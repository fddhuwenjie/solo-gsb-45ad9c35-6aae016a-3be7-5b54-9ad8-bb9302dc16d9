"""圆顶静平衡配重规划：枚举择优与偏差穷举的算法及 HTTP API 测试。"""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner.balance import (
    Slot,
    plan_balance,
    slot_levels,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _slot(sid, radius, az, limit, step=1.0):
    return Slot(
        id=sid, radius=radius, azimuth_deg=az, mass_limit=limit,
        levels=slot_levels(limit, step),
    )


def _payload(parts, slots, step=1.0, mtol=0.0, atol=0.0, limit=0.0):
    return {
        "installed_parts": [
            {"mass": m, "radius": r, "azimuth": a} for (m, r, a) in parts
        ],
        "slots": [
            {"id": i, "radius": r, "azimuth": a, "mass_limit": lim}
            for (i, r, a, lim) in slots
        ],
        "mass_step": step,
        "mass_tolerance_ratio": mtol,
        "azimuth_tolerance": atol,
        "residual_moment_limit": limit,
    }


# ---------------------------------------------------------------------------
# 档位枚举
# ---------------------------------------------------------------------------


def test_slot_levels_basic_and_boundary():
    assert slot_levels(40.0, 1.0) == tuple(float(k) for k in range(1, 41))
    assert slot_levels(2.5, 1.0) == (1.0, 2.0)
    # 恰为整数倍（允许一个舍入误差量级的容差）
    assert slot_levels(3.0, 1.0) == (1.0, 2.0, 3.0)
    assert slot_levels(0.9, 1.0) == ()


# ---------------------------------------------------------------------------
# 标称平衡（零偏差）
# ---------------------------------------------------------------------------


def test_nominal_exact_cancellation_single_slot():
    # 原始矩 10 向北；180° 槽放 10kg（r=1）精确抵消
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], [_slot("A", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=0.0,
    )
    assert len(plan.weights) == 1
    assert plan.weights[0].slot_id == "A"
    assert plan.weights[0].nominal_mass == 10.0
    assert plan.nominal_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.worst_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.compliant is True
    # 零偏差时全部 9 个情景相同：最坏组合取符号字典序最小者 (-1,-1)，
    # 其实际质量/方位仍等于标称（容差为 0）
    combo0 = plan.worst_combo[0]
    assert combo0.mass_sign == -1
    assert combo0.azimuth_sign == -1
    assert combo0.mass == pytest.approx(10.0, abs=1e-12)
    assert combo0.azimuth_deg == pytest.approx(180.0, abs=1e-12)
    assert plan.total_balance_mass == 10.0
    assert plan.evaluated_plans == 41  # 零配重 + 40 个单槽档
    assert plan.max_scenarios_per_plan == 9  # 单配重仍枚举 9 个情景


def test_azimuth_crosses_zero_equivalence():
    # 已装部件在 350°（跨 0° 北侧），反向方位 170° 与 -190° 几何等价：
    # 两个分别标称 170° 与 -190° 的等半径槽，方案结果必须一致。
    common = dict(
        parts=[(10.0, 1.0, 350.0)], mass_step=1.0,
        mass_tolerance_ratio=0.02, azimuth_tolerance_deg=1.0,
        residual_moment_limit=1.0,
    )
    p1 = plan_balance(common["parts"], [_slot("A", 1.0, 170.0, 40.0)],
                      mass_step=1.0, mass_tolerance_ratio=0.02,
                      azimuth_tolerance_deg=1.0, residual_moment_limit=1.0)
    p2 = plan_balance(common["parts"], [_slot("A", 1.0, -190.0, 40.0)],
                      mass_step=1.0, mass_tolerance_ratio=0.02,
                      azimuth_tolerance_deg=1.0, residual_moment_limit=1.0)
    assert p1.weights[0].nominal_mass == p2.weights[0].nominal_mass == 10.0
    assert p1.nominal_magnitude == pytest.approx(p2.nominal_magnitude, abs=1e-9)
    assert p1.worst_magnitude == pytest.approx(p2.worst_magnitude, abs=1e-9)
    # 精确抵消：350° 与 170° 相差 180°
    assert p1.nominal_magnitude == pytest.approx(0.0, abs=1e-12)


def test_original_moment_sums_parts():
    # 两个垂直部件：6 在 0°（北）、8 在 90°（东），合成矩模长 10
    plan = plan_balance(
        [(6.0, 1.0, 0.0), (8.0, 1.0, 90.0)],
        [_slot("A", 1.0, 270.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=100.0,
    )
    assert plan.original_x == pytest.approx(8.0, abs=1e-12)
    assert plan.original_y == pytest.approx(6.0, abs=1e-12)
    assert plan.original_magnitude == pytest.approx(10.0, abs=1e-12)


def test_two_slot_split_nominal_balance():
    # 原始矩 10 向北；170°/190° 对称两槽（step=0.1），东西分量互相抵消。
    # 2 m cos10° 对标 10 → m ≈ 5.077：5.0 档欠 0.1519、5.1 档过 0.0450，
    # 标称最优为 5.1（过补偿，残余沿南向，y=-0.0450）。
    slots = [
        _slot("A", 1.0, 170.0, 40.0, step=0.1),
        _slot("B", 1.0, 190.0, 40.0, step=0.1),
    ]
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], slots,
        mass_step=0.1, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert len(plan.weights) == 2
    ma = next(w for w in plan.weights if w.slot_id == "A").nominal_mass
    mb = next(w for w in plan.weights if w.slot_id == "B").nominal_mass
    assert ma == pytest.approx(mb, abs=1e-12)  # 对称：两槽等质量
    assert ma == pytest.approx(5.1, abs=1e-12)
    assert plan.nominal_x == pytest.approx(0.0, abs=1e-12)
    assert plan.nominal_y == pytest.approx(-0.0450, abs=1e-3)
    assert plan.nominal_magnitude == pytest.approx(0.0450, abs=1e-3)
    assert plan.compliant is True


# ---------------------------------------------------------------------------
# 零配重
# ---------------------------------------------------------------------------


def test_zero_balance_when_original_moment_zero():
    # 两个对消部件 → 原始矩为 0，零配重方案即最优且达标
    plan = plan_balance(
        [(5.0, 1.0, 0.0), (5.0, 1.0, 180.0)],
        [_slot("A", 1.0, 90.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.05,
        azimuth_tolerance_deg=2.0, residual_moment_limit=0.0,
    )
    assert plan.weights == ()
    assert plan.total_balance_mass == 0.0
    assert plan.nominal_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.worst_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.worst_combo == ()
    assert plan.compliant is True
    assert plan.mass_vector == (0.0,)


def test_zero_mass_and_zero_radius_parts_contribute_nothing():
    plan = plan_balance(
        [(0.0, 2.0, 45.0), (7.0, 0.0, 90.0)],
        [_slot("A", 0.0, 10.0, 40.0)],  # 零半径槽放任何配重都无矩
        mass_step=1.0, mass_tolerance_ratio=0.05,
        azimuth_tolerance_deg=1.0, residual_moment_limit=0.0,
    )
    assert plan.original_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.weights == ()
    assert plan.compliant is True


# ---------------------------------------------------------------------------
# 偏差穷举与“偏差改变最优方案”
# ---------------------------------------------------------------------------


def test_deviation_changes_optimal_plan():
    # 原始矩 M0=10 向北；唯一槽 r=1.05、180°、step=1。
    # 标称：10kg 残余 0.5（欠 0.5），9kg 残余 0.55 → 标称最优是 10kg；
    # 质量偏差 ε=0.05、无方位偏差时：
    #   9kg（欠补偿）最坏在 -5%：|10 - 9·1.05·0.95| = 1.0225
    #   10kg（欠补偿）最坏在 -5%：|10 - 10·1.05·0.95| = 0.025；
    #                            在 +5%：|10 - 10·1.05·1.05| = 1.025
    # 10kg 的最坏 1.025 略大于 9kg 的最坏 1.0225 —— 偏差使 9kg 反超，
    # 即偏差改变了最优档位；9kg 的最坏情景是质量负偏差（欠得更多）。
    slots = [_slot("A", 1.05, 180.0, 40.0)]
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], slots,
        mass_step=1.0, mass_tolerance_ratio=0.05,
        azimuth_tolerance_deg=0.0, residual_moment_limit=2.0,
    )
    assert plan.weights[0].nominal_mass == 9.0
    assert plan.nominal_magnitude == pytest.approx(0.55, abs=1e-12)
    assert plan.worst_magnitude == pytest.approx(1.0225, abs=1e-9)
    combo = plan.worst_combo[0]
    assert combo.mass_sign == -1
    assert combo.azimuth_sign == -1  # 方位容差为 0，三情景并列取 -1
    assert combo.mass == pytest.approx(9.0 * 0.95, abs=1e-12)

    # 对照：关掉质量偏差后，标称最优的 10kg 重新胜出
    plan_nominal = plan_balance(
        [(10.0, 1.0, 0.0)], slots,
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=2.0,
    )
    assert plan_nominal.weights[0].nominal_mass == 10.0
    assert plan_nominal.nominal_magnitude == pytest.approx(0.5, abs=1e-12)


def test_worst_scenario_explores_all_mass_and_azimuth_signs():
    # 原始矩 10 向北；槽 180° r=1 放 10kg，标称精确抵消。
    # ε=0.1、δ=2° 时最坏质量偏差为 +10%（过补偿 11kg）；方位 ±2° 对称，
    # 两种方位偏差模长相同，符号并列取 -1（即实际方位 178°）。
    # 残余 = 10@0° + 11@178°：东向 11·sin178°、北向 10 + 11·cos178°。
    exp_x = 11.0 * math.sin(math.radians(178.0))
    exp_y = 10.0 + 11.0 * math.cos(math.radians(178.0))
    expected_worst = math.hypot(exp_x, exp_y)
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], [_slot("A", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.1,
        azimuth_tolerance_deg=2.0, residual_moment_limit=20.0,
    )
    assert plan.weights[0].nominal_mass == 10.0
    assert plan.nominal_magnitude == pytest.approx(0.0, abs=1e-12)
    assert plan.worst_magnitude == pytest.approx(expected_worst, abs=1e-9)
    combo = plan.worst_combo[0]
    assert (combo.mass_sign, combo.azimuth_sign) == (1, -1)
    assert combo.mass == pytest.approx(11.0, abs=1e-12)
    assert combo.azimuth_deg == pytest.approx(178.0, abs=1e-12)
    assert plan.worst_x == pytest.approx(exp_x, abs=1e-9)
    assert plan.worst_y == pytest.approx(exp_y, abs=1e-9)
    # 81 个情景上限仅两槽时达到；单槽方案为 9
    assert plan.max_scenarios_per_plan == 9


def test_worst_scenario_combo_lexicographic_tie():
    # 零方位偏差、质量 ±10% 对称：+10% 与 -10% 的最坏模长相等。
    # 符号字典序 (-1 < +1) 决定最坏组合取负偏差。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], [_slot("A", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.1,
        azimuth_tolerance_deg=0.0, residual_moment_limit=20.0,
    )
    assert plan.weights[0].nominal_mass == 10.0
    assert plan.worst_magnitude == pytest.approx(1.0, abs=1e-12)
    # 方位容差为 0，三个方位情景完全相同；符号字典序最小为 (-1,-1)
    assert plan.worst_combo[0].mass_sign == -1
    assert plan.worst_combo[0].azimuth_sign == -1


def test_noncompliant_returns_best_anyway():
    # 原始矩 10.6、step=1、零偏差：最优档位 11kg 的标称残余 0.4，
    # 限值 0.1 下无任何方案可达标，但仍返回择优最优方案并标记 compliant=false。
    plan = plan_balance(
        [(10.6, 1.0, 0.0)], [_slot("A", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=0.1,
    )
    assert plan.weights[0].nominal_mass == 11.0
    assert plan.compliant is False
    assert plan.worst_magnitude == pytest.approx(0.4, abs=1e-12)


def test_compliance_boundary_within_tie_epsilon():
    # 残余恰为 0（浮点 ~1e-15）与限值 0：差值远小于 1e-9，判达标。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)], [_slot("A", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=0.0,
    )
    assert plan.compliant is True


# ---------------------------------------------------------------------------
# 并列与择优键
# ---------------------------------------------------------------------------


def test_tie_identical_slots_picks_smaller_id():
    # 两个同方位槽半径不同：A(r=1) 用 10kg 精确抵消，B(r=0.5) 需 20kg。
    # 零偏差下两方案最坏/标称模长都为 0（并列），按总质量取 A。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)],
        [_slot("A", 1.0, 180.0, 40.0), _slot("B", 0.5, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert [w.slot_id for w in plan.weights] == ["A"]
    assert plan.mass_vector == (10.0, 0.0)
    assert plan.slot_ids == ("A", "B")


def test_tie_equal_single_slot_choices_prefers_smaller_id_vector():
    # 两个完全相同的槽单独使用时结果一致：(10,0) 与 (0,10) 的前三个键
    # 全并列，按质量向量字典序取 ID 较小的 A（两槽等摊方案总质量相同
    # 但残余在零死区内也并列，不过其向量 (5,5) 与 (10,0) 的首分量差
    # 5 > 1e-9，故 (5,5) 反而字典序更小 —— 见下一分摊用例）。
    # 这里用零半径的 B 排除分摊，只比较单槽选择。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)],
        [_slot("A", 1.0, 180.0, 40.0), _slot("B", 0.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert [w.slot_id for w in plan.weights] == ["A"]
    assert plan.mass_vector == (10.0, 0.0)


def test_tie_split_mass_vector_lexicographic():
    # 两个完全相同的槽、零偏差，单用 A/B 的向量分别为 (10,0)、(0,10)，
    # 两槽各 5kg 为 (5,5)；前三个键（最坏 0、标称 0、总质量 10）全并列，
    # 标准字典序为 (0,10) < (5,5) < (10,0)，故最优为单用末位槽 B。
    # 该结果也验证了“未使用槽位记 0”参与字典序比较。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)],
        [_slot("A", 1.0, 180.0, 40.0), _slot("B", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert [w.slot_id for w in plan.weights] == ["B"]
    assert plan.mass_vector == (0.0, 10.0)
    assert plan.total_balance_mass == 10.0


def test_tie_epsilon_between_scalars_treated_as_tie():
    # A(r=1+5e-12) 用 10kg 时残余仅 5e-11（< 1e-9，视为与 0 并列）；
    # B(r=2) 用 5kg 精确抵消。最坏/标称键全并列后按总配重择优：
    # B 的 5kg 胜过 A 的 10kg（即便 A 的残余在数值上略大）。
    plan = plan_balance(
        [(10.0, 1.0, 0.0)],
        [_slot("A", 1.0 + 5e-12, 180.0, 40.0),
         _slot("B", 2.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert [w.slot_id for w in plan.weights] == ["B"]
    assert plan.total_balance_mass == 5.0
    assert plan.nominal_magnitude == pytest.approx(0.0, abs=1e-12)


def test_tie_epsilon_in_mass_vector():
    # 前三个键完全相同、质量向量首分量相差不超过 1e-9：视为并列，
    # 任意方向都不严格更优。
    from app.planner.balance import _strictly_better

    k_a = (0.0, 0.0, 10.0, (10.0, 0.0))
    # 同一位置的分量差 1e-13 并列；k_b 的第二分量与 k_a 的第二分量
    # (0.0) 差 1.0 > 1e-9，故 k_a 的向量字典序更小——这里只验证
    # “首分量 1e-13 的差不影响序关系”：
    k_b = (0.0, 0.0, 10.0, (10.0 - 1e-13, 1.0))
    assert _strictly_better(k_a, k_b) is True
    assert _strictly_better(k_b, k_a) is False

    # 逐分量都在 1e-9 以内（含零）：向量并列
    k_b2 = (0.0, 0.0, 10.0, (10.0 - 1e-13, 1e-12))
    assert _strictly_better(k_a, k_b2) is False
    assert _strictly_better(k_b2, k_a) is False

    # 最坏键差值 5e-10（< 1e-9）虽并列，但零死区只在两值都贴近零时生效；
    # 此处 incumbent 的最坏为 0、candidate 为 5e-10，5e-10 不贴零，
    # 而 5e-10 - 0 <= 1e-9 仍按差值容差判并列 → 继续比较标称（并列）、
    # 总质量：9 < 10，故 candidate 严格更优。
    k_c = (5e-10, 0.0, 9.0, (9.0, 0.0))
    assert _strictly_better(k_c, k_a) is True
    assert _strictly_better(k_a, k_c) is False

    # 两值都贴近零（≤ 1e-9）时同样并列，由后续键决定：总质量 11 > 10，
    # 故 k_d 严格更差
    k_d = (5e-10, 0.0, 11.0, (11.0, 0.0))
    assert _strictly_better(k_d, k_a) is False
    assert _strictly_better(k_a, k_d) is True

    # 差值超过 1e-9 时正常择优：最坏模长小者优；
    # 最坏并列时标称小者优；再并列时质量向量字典序小者优
    assert _strictly_better((1.0, 9.0, 10.0, (10.0, 0.0)), k_a) is False
    assert _strictly_better((0.0, 0.0, 10.0, (9.0, 1.0)), k_a) is True
    assert _strictly_better((0.0, 0.0, 10.0, (10.0, 2.0)), k_a) is False


def test_prefer_less_total_mass_on_magnitude_tie():
    # A 半径 2、B 半径 1，同在 180°：抵消矩 10 时 A 用 5kg、B 用 10kg，
    # 零偏差下残余与最坏模长都为 0（并列），择优取总质量小的 A
    plan = plan_balance(
        [(10.0, 1.0, 0.0)],
        [_slot("A", 2.0, 180.0, 40.0), _slot("B", 1.0, 180.0, 40.0)],
        mass_step=1.0, mass_tolerance_ratio=0.0,
        azimuth_tolerance_deg=0.0, residual_moment_limit=1.0,
    )
    assert [w.slot_id for w in plan.weights] == ["A"]
    assert plan.total_balance_mass == 5.0
    assert plan.worst_magnitude == pytest.approx(0.0, abs=1e-12)


def test_slots_input_order_does_not_change_result():
    slots_ab = [_slot("A", 1.0, 170.0, 20.0), _slot("B", 1.0, 190.0, 20.0)]
    slots_ba = [slots_ab[1], slots_ab[0]]
    p1 = plan_balance(
        [(10.0, 1.0, 0.0)], slots_ab, 1.0, 0.02, 1.0, 2.0)
    p2 = plan_balance(
        [(10.0, 1.0, 0.0)], slots_ba, 1.0, 0.02, 1.0, 2.0)
    # 结果与输入顺序无关：ID 排序、质量向量、各模长完全一致
    assert [w.slot_id for w in p1.weights] == [w.slot_id for w in p2.weights]
    assert p1.slot_ids == p2.slot_ids == ("A", "B")
    assert p1.mass_vector == p2.mass_vector
    assert p1.nominal_magnitude == pytest.approx(p2.nominal_magnitude, abs=1e-12)
    assert p1.worst_magnitude == pytest.approx(p2.worst_magnitude, abs=1e-12)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_api_nominal_exact_balance():
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(10.0, 1.0, 0.0)],
        slots=[("A", 1.0, 180.0, 40.0)],
        mtol=0.0, atol=0.0, limit=0.0,
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_moment"] == {"east": 0.0, "north": 10.0}
    assert body["original_magnitude"] == pytest.approx(10.0, abs=1e-9)
    assert len(body["weights"]) == 1
    w = body["weights"][0]
    assert w["slot_id"] == "A"
    assert w["nominal_mass"] == 10.0
    assert w["nominal_azimuth"] == 180.0
    assert body["nominal_residual"]["east"] == pytest.approx(0.0, abs=1e-9)
    assert body["nominal_residual"]["north"] == pytest.approx(0.0, abs=1e-9)
    assert body["nominal_magnitude"] == pytest.approx(0.0, abs=1e-9)
    assert body["worst_magnitude"] == pytest.approx(0.0, abs=1e-9)
    # 零偏差下 9 个情景相同，取符号字典序最小的 (-1,-1)
    assert body["worst_deviation_combo"][0]["mass_sign"] == -1
    assert body["worst_deviation_combo"][0]["azimuth_sign"] == -1
    assert body["compliant"] is True
    assert body["details"]["evaluated_plans"] == 41
    assert body["details"]["max_scenarios_per_plan"] == 9
    assert body["details"]["slot_ids"] == ["A"]
    assert body["details"]["mass_vector"] == [10.0]


def test_api_crosses_zero():
    # 已装部件在 -5°（等价 355°），反向槽 175°
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(8.0, 1.0, -5.0)],
        slots=[("s", 1.0, 175.0, 40.0), ("t", 1.0, -185.0, 40.0)],
        mtol=0.01, atol=0.5, limit=1.0,
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    # 8kg 精确抵消（-5° 与 175° 差 180°）
    assert body["weights"][0]["nominal_mass"] == 8.0
    assert body["nominal_magnitude"] == pytest.approx(0.0, abs=1e-9)
    assert body["compliant"] is True


def test_api_zero_balance():
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(3.0, 1.0, 90.0), (3.0, 1.0, 270.0)],
        slots=[("A", 1.0, 0.0, 40.0)],
        mtol=0.1, atol=2.0, limit=0.0,
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["weights"] == []
    assert body["worst_deviation_combo"] == []
    assert body["total_balance_mass"] == 0.0
    assert body["worst_magnitude"] == pytest.approx(0.0, abs=1e-9)
    assert body["compliant"] is True
    assert body["details"]["mass_vector"] == [0.0]


def test_api_deviation_changes_optimal():
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(10.0, 1.0, 0.0)],
        slots=[("A", 1.05, 180.0, 40.0)],
        mtol=0.05, atol=0.0, limit=2.0,
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["weights"][0]["nominal_mass"] == 9.0
    assert body["worst_magnitude"] == pytest.approx(1.0225, abs=1e-9)
    # 9kg 欠补偿，最坏情景是质量再 -5%（方位容差为 0，符号并列取 -1）
    assert body["worst_deviation_combo"][0]["mass_sign"] == -1
    assert body["compliant"] is True


def test_api_two_slots_eighty_one_scenarios():
    # 8 槽 40 档规模过大不放进 HTTP 回归；两槽时 max_scenarios=81
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(12.3, 0.7, 63.0)],
        slots=[("A", 1.0, 240.0, 20.0), ("B", 1.2, 246.0, 20.0)],
        mtol=0.03, atol=1.5, limit=5.0,
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["details"]["max_scenarios_per_plan"] == 81
    # 最坏组合条数等于非零配重数，每条符号 ∈ {-1,0,1}
    for c in body["worst_deviation_combo"]:
        assert c["mass_sign"] in (-1, 0, 1)
        assert c["azimuth_sign"] in (-1, 0, 1)
    # 向量一致性：标称残余 ≈ 原始矩 + Σ配重标称矩
    east = body["original_moment"]["east"]
    north = body["original_moment"]["north"]
    for w in body["weights"]:
        a = math.radians(w["nominal_azimuth"])
        m = w["nominal_mass"] * w["radius"]
        east += m * math.sin(a)
        north += m * math.cos(a)
    assert body["nominal_residual"]["east"] == pytest.approx(east, abs=1e-7)
    assert body["nominal_residual"]["north"] == pytest.approx(north, abs=1e-7)
    assert body["nominal_magnitude"] == pytest.approx(
        math.hypot(body["nominal_residual"]["east"],
                   body["nominal_residual"]["north"]),
        abs=1e-7,
    )


def test_api_validation_errors():
    base = _payload(
        parts=[(10.0, 1.0, 0.0)],
        slots=[("A", 1.0, 180.0, 40.0)],
        mtol=0.0, atol=0.0, limit=0.0,
    )

    # 槽位 ID 重复
    bad = _payload(parts=[(1.0, 1.0, 0.0)],
                   slots=[("A", 1.0, 0.0, 10.0), ("A", 1.0, 90.0, 10.0)])
    assert client.post("/api/v1/plan-dome-balance", json=bad).status_code == 422

    # 槽位数超过 8
    bad = base | {"slots": [
        {"id": f"s{i}", "radius": 1.0, "azimuth": 0.0, "mass_limit": 10.0}
        for i in range(9)
    ]}
    assert client.post("/api/v1/plan-dome-balance", json=bad).status_code == 422

    # 非零档位超过 40
    bad = base | {"mass_step": 1.0, "slots": [
        {"id": "A", "radius": 1.0, "azimuth": 180.0, "mass_limit": 41.0}
    ]}
    assert client.post("/api/v1/plan-dome-balance", json=bad).status_code == 422

    # mass_limit < mass_step（一个档位都没有）
    bad = base | {"slots": [
        {"id": "A", "radius": 1.0, "azimuth": 180.0, "mass_limit": 0.5}
    ]}
    assert client.post("/api/v1/plan-dome-balance", json=bad).status_code == 422

    # 非法偏差参数
    for patch in (
        {"mass_tolerance_ratio": 1.0},
        {"mass_tolerance_ratio": -0.1},
        {"azimuth_tolerance": -1.0},
        {"residual_moment_limit": -0.1},
        {"mass_step": 0.0},
    ):
        assert client.post(
            "/api/v1/plan-dome-balance", json=base | patch
        ).status_code == 422

    # 负质量 / 负半径 / 空部件清单 / 空 ID / NaN
    assert client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(-1.0, 1.0, 0.0)], slots=[("A", 1.0, 180.0, 40.0)],
    )).status_code == 422
    assert client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(1.0, -1.0, 0.0)], slots=[("A", 1.0, 180.0, 40.0)],
    )).status_code == 422
    no_parts = base | {"installed_parts": []}
    assert client.post("/api/v1/plan-dome-balance", json=no_parts).status_code == 422
    bad_id = base | {"slots": [
        {"id": "", "radius": 1.0, "azimuth": 180.0, "mass_limit": 40.0}
    ]}
    assert client.post("/api/v1/plan-dome-balance", json=bad_id).status_code == 422


def test_api_rejects_non_finite_inputs_at_model_layer():
    # NaN/Infinity 由 Pydantic 的 allow_inf_nan=False 在模型层拒绝。
    # （不经 HTTP 发送：Starlette 的 422 错误体会原样回显 NaN，
    # 其默认 JSON 渲染器本身无法序列化 NaN，属于框架行为。）
    import math

    from pydantic import ValidationError

    from app.models import PlanDomeBalanceRequest

    valid = {
        "installed_parts": [{"mass": 10.0, "radius": 1.0, "azimuth": 0.0}],
        "slots": [{"id": "A", "radius": 1.0, "azimuth": 180.0,
                   "mass_limit": 40.0}],
        "mass_step": 1.0,
        "mass_tolerance_ratio": 0.0,
        "azimuth_tolerance": 0.0,
        "residual_moment_limit": 0.0,
    }
    for path, value in [
        (("residual_moment_limit",), float("nan")),
        (("residual_moment_limit",), math.inf),
        (("mass_step",), float("nan")),
        (("installed_parts", 0, "mass"), math.inf),
        (("slots", 0, "azimuth"), float("nan")),
    ]:
        body = {**valid}
        target = body
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValidationError):
            PlanDomeBalanceRequest.model_validate(body)


def test_api_azimuth_returned_unnormalized():
    # 请求方位用 -190° 表示时，回显保持 -190°（不强制归一化到 [0,360)）
    r = client.post("/api/v1/plan-dome-balance", json=_payload(
        parts=[(5.0, 1.0, 350.0)],
        slots=[("A", 1.0, -190.0, 40.0)],
        mtol=0.02, atol=1.0, limit=1.0,
    ))
    assert r.status_code == 200, r.text
    w = r.json()["weights"][0]
    assert w["nominal_azimuth"] == -190.0
    assert w["worst_azimuth"] in (-191.0, -190.0, -189.0)


# ---------------------------------------------------------------------------
# 独立暴力对拍：随机小实例上枚举全部 ≤2 槽方案与全部偏差情景
# ---------------------------------------------------------------------------


def _brute_force(parts, slots, step, mtol, atol, limit):
    """与规划器无关的直接枚举实现，返回规划器响应应满足的全部关键量。"""

    def vec(m, r, a):
        ar = math.radians(a)
        return m * r * math.sin(ar), m * r * math.cos(ar)

    ox = sum(m * r * math.sin(math.radians(a)) for m, r, a in parts)
    oy = sum(m * r * math.cos(math.radians(a)) for m, r, a in parts)

    levels_per = [slot_levels(s[3], step) for s in slots]
    ids = [s[0] for s in slots]

    candidates = [[]]  # 零配重
    for i in range(len(slots)):
        for li, m in enumerate(levels_per[i]):
            candidates.append([(i, li)])
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            for li, mi in enumerate(levels_per[i]):
                for lj, mj in enumerate(levels_per[j]):
                    candidates.append([(i, li), (j, lj)])

    best = None
    for cand in candidates:
        nx, ny, total = ox, oy, 0.0
        mv = [0.0] * len(slots)
        for i, li in cand:
            m = levels_per[i][li]
            vx, vy = vec(m, slots[i][1], slots[i][2])
            nx += vx
            ny += vy
            total += m
            mv[i] = m
        n_mag = math.hypot(nx, ny)

        # 偏差情景（独立三重积实现），代码 (sm0,sa0,sm1,sa1,...)
        w_mag = -1.0
        w_combo = None
        signs = (-1, 0, 1)
        combos = [()]
        for _ in cand:
            combos = [c + (sm, sa) for c in combos for sm in signs for sa in signs]
        for code in combos:
            rx, ry = ox, oy
            for pos, (i, li) in enumerate(cand):
                sm, sa = code[2 * pos], code[2 * pos + 1]
                mp = levels_per[i][li] * (1 + sm * mtol)
                ap = slots[i][2] + sa * atol
                vx, vy = vec(mp, slots[i][1], ap)
                rx += vx
                ry += vy
            mag = math.hypot(rx, ry)
            if w_mag < 0 or mag - w_mag > 1e-9:
                w_mag = mag
                w_combo = code

        key = (w_mag, n_mag, total, tuple(mv))
        if best is None:
            best = (key, cand, w_mag, w_combo)
        else:
            bk = best[0]
            less = False
            for t in range(4):
                if t < 3:
                    if abs(key[t] - bk[t]) <= 1e-9:
                        continue
                    less = key[t] < bk[t]
                    break
                else:
                    less = False
                    for x, y in zip(key[3], bk[3]):
                        if abs(x - y) <= 1e-9:
                            continue
                        less = x < y
                        break
            if less:
                best = (key, cand, w_mag, w_combo)

    key, chosen, w_mag, w_combo = best
    return {
        "ids": ids,
        "original": (ox, oy, math.hypot(ox, oy)),
        "chosen": [(slots[i][0], levels_per[i][li]) for i, li in chosen],
        "total": key[2],
        "nominal_mag": key[1],
        "worst_mag": key[0],
        "mass_vector": key[3],
        "compliant": not (w_mag > limit and w_mag - limit > 1e-9),
        "n_candidates": len(candidates),
    }


def test_brute_force_cross_check_random():
    import random

    rng = random.Random(20260929)
    for trial in range(24):
        n_slots = rng.randint(1, 4)
        step = rng.choice([0.5, 1.0, 2.0])
        slots = []
        used_az = []
        for i in range(n_slots):
            limit = rng.randint(1, 6) * step
            az = rng.choice([0.0, 30.0, 90.0, 170.0, 180.0, 190.0,
                             270.0, 350.0, -10.0])
            slots.append((f"s{i}", rng.choice([0.5, 1.0, 1.5, 2.0]),
                          az, limit))
        parts = [
            (rng.uniform(0.0, 12.0), rng.choice([0.5, 1.0, 2.0]),
             rng.choice([0.0, 60.0, 180.0, 270.0, 350.0]))
            for _ in range(rng.randint(1, 3))
        ]
        mtol = rng.choice([0.0, 0.02, 0.05])
        atol = rng.choice([0.0, 1.0, 3.0])
        limit = rng.choice([0.0, 0.5, 2.0])

        expected = _brute_force(parts, slots, step, mtol, atol, limit)
        plan = plan_balance(
            parts,
            [Slot(sid, r, a, lim, slot_levels(lim, step))
             for sid, r, a, lim in slots],
            mass_step=step, mass_tolerance_ratio=mtol,
            azimuth_tolerance_deg=atol, residual_moment_limit=limit,
        )

        assert plan.evaluated_plans == expected["n_candidates"]
        assert plan.slot_ids == tuple(sorted(s[0] for s in slots))
        got_chosen = sorted((w.slot_id, w.nominal_mass) for w in plan.weights)
        assert got_chosen == sorted(expected["chosen"])
        assert plan.total_balance_mass == pytest.approx(expected["total"], abs=1e-9)
        assert plan.nominal_magnitude == pytest.approx(
            expected["nominal_mag"], abs=1e-9)
        assert plan.worst_magnitude == pytest.approx(
            expected["worst_mag"], abs=1e-9)
        assert plan.compliant == expected["compliant"]
        for a, b in zip(plan.mass_vector, expected["mass_vector"]):
            assert a == pytest.approx(b, abs=1e-9)
        assert plan.original_x == pytest.approx(expected["original"][0], abs=1e-9)
        assert plan.original_y == pytest.approx(expected["original"][1], abs=1e-9)
        assert plan.original_magnitude == pytest.approx(
            expected["original"][2], abs=1e-9)
