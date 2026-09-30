"""圆顶检修配重换装顺序规划：状态搜索、并列择优与无路诊断的算法/HTTP 测试。"""

import math
import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner.resequence import (
    WeightSequenceValidationError,
    plan_weight_sequence,
)

client = TestClient(app)

EPS = 1e-9


# ---------------------------------------------------------------------------
# 独立参考实现：与规划器无关的小规模全状态 BFS
# ---------------------------------------------------------------------------


def _vec(m, r, a):
    ar = math.radians(a)
    return m * r * math.sin(ar), m * r * math.cos(ar)


def _excess(v, lim):
    return v - lim if v - lim > EPS else 0.0


def reference_plan(fixed, weights, current, target, mass_lim, mom_lim):
    """暴力参考：全掩码 BFS，标签 (最大残余模长, 动作序列) 与产品一致。

    返回 dict：feasible / length / max_mag / actions / reachable / blocked。
    实现刻意与产品不同：直接在 while 循环里松弛直到本层无更新。
    """

    weights = sorted(weights, key=lambda w: w[0])
    n = len(weights)
    fm = sum(p[0] for p in fixed)
    fx = sum(_vec(*p)[0] for p in fixed)
    fy = sum(_vec(*p)[1] for p in fixed)

    nmask = 1 << n

    def phys(mask):
        m = fm
        x, y = fx, fy
        for k, (wid, wm, wr, wa) in enumerate(weights):
            if mask & (1 << k):
                vx, vy = _vec(wm, wr, wa)
                x += vx
                y += vy
                m += wm
        return m, x, y, math.hypot(x, y)

    legal = [False] * nmask
    for mask in range(nmask):
        m, _x, _y, mag = phys(mask)
        legal[mask] = (
            _excess(m, mass_lim) == 0.0 and _excess(mag, mom_lim) == 0.0
        )

    start = sum(1 << k for k, w in enumerate(weights) if w[0] in current)
    goal = sum(1 << k for k, w in enumerate(weights) if w[0] in target)
    if not (legal[start] and legal[goal]):
        return {"feasible": None, "legal_start": legal[start],
                "legal_goal": legal[goal]}

    dist = {start: 0}
    # label: (max_mag, seq)，seq 项为 (action, id)
    label = {start: (phys(start)[3], ())}
    frontier = [start]
    d = 0
    while frontier:
        nxt = {}
        for mask in frontier:
            cmax, cseq = label[mask]
            for k, (wid, wm, wr, wa) in enumerate(weights):
                bit = 1 << k
                if mask & bit:
                    act, nm = "remove", mask ^ bit
                else:
                    act, nm = "install", mask | bit
                if not legal[nm] or nm in dist:
                    continue
                mm = max(cmax, phys(nm)[3])
                cand = (mm, cseq + ((act, wid),))
                if nm not in nxt or _ref_label_less(cand, nxt[nm]):
                    nxt[nm] = cand
        for nm, lab in nxt.items():
            dist[nm] = d + 1
            label[nm] = lab
        frontier = list(nxt)
        d += 1

    out = {
        "reachable": len(dist),
        "feasible": goal in dist,
        "start": start,
        "goal": goal,
        "weights": weights,
        "phys": phys,
        "legal": legal,
        "dist": dist,
    }
    if goal in dist:
        out["length"] = len(label[goal][1])
        out["max_mag"] = label[goal][0]
        out["actions"] = label[goal][1]
    else:
        blocked = []
        for mask in dist:
            for k, (wid, wm, wr, wa) in enumerate(weights):
                bit = 1 << k
                if mask & bit:
                    act, nm = "remove", mask ^ bit
                else:
                    act, nm = "install", mask | bit
                if legal[nm]:
                    continue
                _m, _x, _y, mag = phys(nm)
                blocked.append((
                    _excess(mag, mom_lim),
                    _excess(phys(nm)[0], mass_lim),
                    act,
                    wid,
                    dist[mask],
                ))
        blocked.sort()
        out["blocked_key"] = blocked[0]
    return out


def _ref_label_less(a, b):
    if abs(a[0] - b[0]) > EPS:
        return a[0] < b[0]
    return a[1] < b[1]


def _ids(mask, weights):
    return tuple(weights[k][0] for k in range(len(weights)) if mask & (1 << k))


# ---------------------------------------------------------------------------
# 基本有解路径
# ---------------------------------------------------------------------------


def test_simple_install_two_weights():
    plan = plan_weight_sequence(
        fixed_parts=[],
        weight_items=[
            ("A", 1.0, 1.0, 0.0),
            ("B", 1.0, 1.0, 90.0),
        ],
        current_ids=set(),
        target_ids={"A", "B"},
        installed_mass_limit=10.0,
        residual_moment_limit=10.0,
    )
    assert plan.feasible is True
    assert len(plan.steps) == 2
    assert [(s.action, s.weight_id) for s in plan.steps] == [
        ("install", "A"), ("install", "B")
    ]
    assert plan.steps[0].installed_ids == ("A",)
    assert plan.steps[1].installed_ids == ("A", "B")
    assert plan.temporary_weight_ids == ()
    # 逐步物理量
    assert plan.steps[0].residual_north == pytest.approx(1.0, abs=1e-9)
    assert plan.steps[1].residual_east == pytest.approx(1.0, abs=1e-9)
    assert plan.steps[1].residual_north == pytest.approx(1.0, abs=1e-9)
    assert plan.steps[1].residual_magnitude == pytest.approx(math.sqrt(2), abs=1e-9)


def test_zero_steps_when_current_equals_target():
    plan = plan_weight_sequence(
        fixed_parts=[(3.0, 1.0, 45.0)],
        weight_items=[("A", 1.0, 1.0, 0.0), ("B", 1.0, 1.0, 90.0)],
        current_ids={"A"},
        target_ids={"A"},
        installed_mass_limit=10.0,
        residual_moment_limit=10.0,
    )
    assert plan.feasible is True
    assert plan.steps == ()
    assert plan.temporary_weight_ids == ()
    assert plan.max_intermediate_magnitude == pytest.approx(
        plan.initial.residual_magnitude, abs=1e-12
    )
    assert plan.reachable_states >= 1


def test_fixed_parts_always_count_in_mass():
    # 固定件 8kg + 配重 3kg = 11，质量上限 10：目标状态越限 → 422
    with pytest.raises(WeightSequenceValidationError) as ei:
        plan_weight_sequence(
            fixed_parts=[(8.0, 0.0, 0.0)],  # 零半径不贡献矩
            weight_items=[("A", 3.0, 1.0, 0.0)],
            current_ids=set(),
            target_ids={"A"},
            installed_mass_limit=10.0,
            residual_moment_limit=10.0,
        )
    assert ei.value.payload["reason"] == "target_state_violates_limits"
    assert ei.value.payload["mass_excess"] == pytest.approx(1.0, abs=1e-9)
    assert ei.value.payload["moment_excess"] == 0.0


def test_moment_convention_north_zero_east_positive():
    # 90° 方位的配重矩应落在东向（x>0, y=0）
    plan = plan_weight_sequence(
        fixed_parts=[],
        weight_items=[("A", 2.0, 3.0, 90.0)],
        current_ids=set(), target_ids={"A"},
        installed_mass_limit=100.0, residual_moment_limit=100.0,
    )
    s = plan.steps[0]
    assert s.residual_east == pytest.approx(6.0, abs=1e-9)
    assert abs(s.residual_north) <= 1e-9
    assert s.residual_magnitude == pytest.approx(6.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 临时配重：借来搭桥、最终拆除
# ---------------------------------------------------------------------------


def test_temporary_weight_bridge_and_removal():
    # 固定 10@0；A=5,r2@180 当前；目标 B=5,r2@150 + C=5,r2@210；
    # 限值 9：直接装 B 残余模长 10 越限；临时 T=2,r4@180（矩 8）搭桥。
    weights = [
        ("A", 5.0, 2.0, 180.0),
        ("B", 5.0, 2.0, 150.0),
        ("C", 5.0, 2.0, 210.0),
        ("T", 2.0, 4.0, 180.0),
    ]
    plan = plan_weight_sequence(
        fixed_parts=[(10.0, 1.0, 0.0)],
        weight_items=weights,
        current_ids={"A"}, target_ids={"B", "C"},
        installed_mass_limit=20.0, residual_moment_limit=9.0,
    )
    assert plan.feasible is True
    assert plan.steps[-1].installed_ids == ("B", "C")
    assert plan.temporary_weight_ids == ("T",)
    # T 的两次操作标记 temporary；换装对象 A 不是临时配重
    touched = {s.weight_id: s.temporary for s in plan.steps}
    assert touched["T"] is True
    assert touched["A"] is False
    # 每一步均不越限
    for s in plan.steps:
        assert s.total_mass <= 20.0 + EPS
        assert s.residual_magnitude <= 9.0 + EPS
    # 无 T 时无路
    plan_no_t = plan_weight_sequence(
        fixed_parts=[(10.0, 1.0, 0.0)],
        weight_items=weights[:3],
        current_ids={"A"}, target_ids={"B", "C"},
        installed_mass_limit=20.0, residual_moment_limit=9.0,
    )
    assert plan_no_t.feasible is False
    assert plan_no_t.reachable_states == 1
    b = plan_no_t.nearest_blocked_action
    # 从起点出发装 B 或装 C：模长 10，超限量 1；字典序 install < remove，
    # install B < install C
    assert (b.action, b.weight_id) == ("install", "B")
    assert b.moment_excess == pytest.approx(1.0, abs=1e-9)
    assert b.mass_excess == 0.0
    assert b.from_depth == 0


def test_old_weight_removed_is_not_temporary():
    # 起点 A、目标空：A 是换装对象而非临时配重
    plan = plan_weight_sequence(
        fixed_parts=[(1.0, 0.0, 0.0)],
        weight_items=[("A", 1.0, 1.0, 0.0)],
        current_ids={"A"}, target_ids=set(),
        installed_mass_limit=10.0, residual_moment_limit=10.0,
    )
    assert plan.feasible is True
    assert [s.action for s in plan.steps] == ["remove"]
    assert plan.temporary_weight_ids == ()
    assert plan.steps[0].temporary is False


# ---------------------------------------------------------------------------
# 最短性与并列择优
# ---------------------------------------------------------------------------


def test_lexicographic_install_before_remove():
    # 起点 {X}，目标 {Y}；在宽松限值下最短长度 2。
    # 候选：install Y → remove X（install 在前）；remove X → install Y。
    # 两条路径中间残余模长完全相同（同向 1kg：1→2 与 1→0 的中间模长
    # 分别为 2 和 1——为使残余键并列，X/Y 取相反方位：
    # X=0°、Y=180°，r=1、m=1）：
    #   install Y 后 {X,Y} 残余 0；remove X 后 {Y} 残余 1（Y@180）；
    #   remove X 后 {} 残余 0；install Y 后 {Y} 残余 1。
    # 中间模长 0 vs 0 并列 → 动作字典序 install < remove，取先装后拆。
    plan = plan_weight_sequence(
        fixed_parts=[],
        weight_items=[
            ("X", 1.0, 1.0, 0.0),
            ("Y", 1.0, 1.0, 180.0),
        ],
        current_ids={"X"}, target_ids={"Y"},
        installed_mass_limit=10.0, residual_moment_limit=10.0,
    )
    assert [(s.action, s.weight_id) for s in plan.steps] == [
        ("install", "Y"), ("remove", "X")
    ]


def test_tie_break_max_residual_distinguishes_paths():
    # 起点 {a}，目标 {a,b,c}，最短 2 步（先装 b 或先装 c）。
    # 方位约定：北 0°、东 90°、南 180°、西 270°。
    # a=5@0（北, (0,5)）；b=4@90（东, (+4,0)）；c=4@270（西, (-4,0)）。
    # 末态 {a,b,c}=(0,5)，模 5（东西分量抵消）。
    #   先 b（东）：{a,b}=(4,5) 模 sqrt(41)≈6.403；再 c → 5；
    #   先 c（西）：{a,c}=(-4,5) 模同为 6.403…… 也并列。
    # 改用不对称质量：b=4@90（东）、c=3@270（西）：
    # 末态 (1,5)，模 sqrt(26)≈5.099。
    #   先 b：(4,5) 模 6.403，max=6.403；
    #   先 c：(-3,5) 模 sqrt(34)≈5.831，max=5.831。
    # 先 c 最大残余更小；而动作字典序 install b 占优 → 验证残余键优先。
    plan = plan_weight_sequence(
        fixed_parts=[],
        weight_items=[
            ("a", 5.0, 1.0, 0.0),
            ("b", 4.0, 1.0, 90.0),
            ("c", 3.0, 1.0, 270.0),
        ],
        current_ids={"a"}, target_ids={"a", "b", "c"},
        installed_mass_limit=20.0, residual_moment_limit=20.0,
    )
    assert [(s.action, s.weight_id) for s in plan.steps] == [
        ("install", "c"), ("install", "b")
    ]
    # 先 c 路径最大残余在中间状态 {a,c}：sqrt(34)（末态为 sqrt(26)）
    assert plan.max_intermediate_magnitude == pytest.approx(
        math.sqrt(34.0), abs=1e-9
    )


def test_action_sequence_lexicographic_after_equal_magnitude_key():
    # 中间残余模长完全并列时由动作序列字典序裁决：
    # a、b 等质量同方位，两种顺序键相同，取 install a 在前。
    plan = plan_weight_sequence(
        fixed_parts=[],
        weight_items=[
            ("a", 1.0, 1.0, 90.0),
            ("b", 1.0, 1.0, 90.0),
        ],
        current_ids=set(), target_ids={"a", "b"},
        installed_mass_limit=10.0, residual_moment_limit=10.0,
    )
    assert [(s.action, s.weight_id) for s in plan.steps] == [
        ("install", "a"), ("install", "b")
    ]


def test_request_weight_order_does_not_change_result():
    weights = [
        ("A", 5.0, 2.0, 180.0),
        ("B", 5.0, 2.0, 150.0),
        ("C", 5.0, 2.0, 210.0),
        ("T", 2.0, 4.0, 180.0),
    ]
    kw = dict(
        fixed_parts=[(10.0, 1.0, 0.0)],
        current_ids={"A"}, target_ids={"B", "C"},
        installed_mass_limit=20.0, residual_moment_limit=9.0,
    )
    p1 = plan_weight_sequence(weight_items=weights, **kw)
    p2 = plan_weight_sequence(weight_items=list(reversed(weights)), **kw)
    seq1 = [(s.action, s.weight_id) for s in p1.steps]
    seq2 = [(s.action, s.weight_id) for s in p2.steps]
    assert seq1 == seq2
    assert p1.max_intermediate_magnitude == pytest.approx(
        p2.max_intermediate_magnitude, abs=1e-12
    )
    assert p1.reachable_states == p2.reachable_states


# ---------------------------------------------------------------------------
# 无路诊断
# ---------------------------------------------------------------------------


def test_infeasible_reports_reachable_count_and_nearest_blocked():
    # 固定 10@0；A=5,r2@180 当前；B,C=2.5,r2@170/190 目标；限值 5。
    # 可达状态：{A}、{A,B}、{A,C}（共 3）；目标 {B,C} 合法但不可达。
    weights = [
        ("A", 5.0, 2.0, 180.0),
        ("B", 2.5, 2.0, 170.0),
        ("C", 2.5, 2.0, 190.0),
    ]
    plan = plan_weight_sequence(
        fixed_parts=[(10.0, 1.0, 0.0)],
        weight_items=weights,
        current_ids={"A"}, target_ids={"B", "C"},
        installed_mass_limit=20.0, residual_moment_limit=5.0,
    )
    assert plan.feasible is False
    assert plan.reachable_states == 3
    assert plan.steps == ()
    b = plan.nearest_blocked_action
    # {A,B} 拆 A → {B}：残余模 5.149682，矩超限量 ≈ 0.149682（最小）
    assert (b.action, b.weight_id) == ("remove", "A")
    assert b.from_depth == 1
    assert b.from_installed_ids == ("A", "B")
    assert b.resulting_installed_ids == ("B",)
    assert b.moment_excess == pytest.approx(0.149682, abs=1e-6)
    assert b.mass_excess == 0.0
    assert b.resulting_residual_magnitude == pytest.approx(5.149682, abs=1e-6)


def test_nearest_blocked_can_be_mass_excess_only():
    # 无路场景中，让“最近一步”仅由总质量超限量决定。
    #
    # 固定件 10kg、矩 10@0；A=5kg r2@180（矩 10@180，抵消固定矩）；
    # B=2kg 零矩。质量上限 16.9（{A,B}=17 超限 0.1），矩限值 5。
    # 状态合法性：
    #   {A}   ：质量 15、矩 0           合法（起点）
    #   {A,B} ：质量 17 > 16.9         仅超质量 0.1，阻断
    #   {B}   ：质量 12、矩 10 > 5     目标合法？否——矩越限。
    # 因此把 B 放在 180° 且 r=2.5（矩 5@180）：
    #   {B}   ：矩 |10@0+5@180| = 5、质量 12 → 合法（目标）
    #   {A,B} ：矩 5@180（A 抵消固定件后只剩 B 矩）、质量 17 → 仅超质量
    #   {}    ：矩 10 越限
    # 从 {A} 出发：装 B 仅超质量阻断；拆 A → {} 矩超限量 5。
    # 故最近阻断动作 = install B（矩超 0、质量超 0.1 < 矩超 5）。
    plan = plan_weight_sequence(
        fixed_parts=[(10.0, 1.0, 0.0)],
        weight_items=[
            ("A", 5.0, 2.0, 180.0),
            ("B", 2.0, 2.5, 180.0),
        ],
        current_ids={"A"}, target_ids={"B"},
        installed_mass_limit=16.9, residual_moment_limit=5.0,
    )
    assert plan.feasible is False
    assert plan.reachable_states == 1  # 装 B 即超质量，{} 越矩
    b = plan.nearest_blocked_action
    assert (b.action, b.weight_id) == ("install", "B")
    assert b.from_depth == 0
    assert b.moment_excess == 0.0
    assert b.mass_excess == pytest.approx(0.1, abs=1e-9)
    assert b.resulting_residual_magnitude == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 422：起点/终点越限
# ---------------------------------------------------------------------------


def test_current_and_target_violation_examples():
    # 固定 10@0、A 矩为 0（半径 0）：起点 {A} 残余模 10，限值 5 → 起点越限
    with pytest.raises(WeightSequenceValidationError) as ei:
        plan_weight_sequence(
            fixed_parts=[(10.0, 1.0, 0.0)],
            weight_items=[("A", 1.0, 0.0, 0.0)],
            current_ids={"A"}, target_ids=set(),
            installed_mass_limit=20.0, residual_moment_limit=5.0,
        )
    p = ei.value.payload
    assert p["reason"] == "current_state_violates_limits"
    assert p["moment_excess"] == pytest.approx(5.0, abs=1e-9)
    assert p["residual"]["north"] == pytest.approx(10.0, abs=1e-9)

    # 起点合法（C=10kg r1@180 抵消固定矩），目标换成零矩配重 A：
    # {A} 残余模 10 越限 → 终点越限（起点 {} 不含 C 时也越限，故保留 C）
    with pytest.raises(WeightSequenceValidationError) as ei:
        plan_weight_sequence(
            fixed_parts=[(10.0, 1.0, 0.0)],
            weight_items=[
                ("C", 10.0, 1.0, 180.0),
                ("A", 1.0, 0.0, 0.0),
            ],
            current_ids={"C"}, target_ids={"A"},
            installed_mass_limit=20.0, residual_moment_limit=5.0,
        )
    p = ei.value.payload
    assert p["reason"] == "target_state_violates_limits"
    assert p["endpoint"] == "target"
    assert p["moment_excess"] == pytest.approx(5.0, abs=1e-9)


def test_boundary_within_epsilon_is_legal():
    # 残余恰为限值（浮点残差级）视为合法，且零步可达
    plan = plan_weight_sequence(
        fixed_parts=[(5.0, 1.0, 0.0)],
        weight_items=[("A", 5.0, 1.0, 180.0)],
        current_ids={"A"}, target_ids={"A"},
        installed_mass_limit=10.0, residual_moment_limit=0.0,
    )
    assert plan.feasible is True
    assert plan.steps == ()


def test_empty_weight_catalog_zero_steps():
    # 无可操作配重：唯一状态即空掩码，起终点都为空时零步可达
    plan = plan_weight_sequence(
        fixed_parts=[(1.0, 1.0, 0.0)],
        weight_items=[],
        current_ids=set(), target_ids=set(),
        installed_mass_limit=10.0, residual_moment_limit=10.0,
    )
    assert plan.feasible is True
    assert plan.steps == ()
    assert plan.reachable_states == 1
    assert plan.explored_states == 1
    assert plan.nearest_blocked_action is None


def test_every_step_respects_both_limits_in_random_like_case():
    # 全在位合法起点：固定 10@0 与 A=5,r2@180（矩 10 向南）抵消，
    # 再加零矩的 D（半径 0）不改变质量矩，起点残余 0、质量 16。
    # 目标 {A}（质量 15、矩 0）：最短一步即拆 D，每步两限值都不越界。
    weights = [
        ("A", 5.0, 2.0, 180.0),
        ("D", 1.0, 0.0, 0.0),
    ]
    plan = plan_weight_sequence(
        fixed_parts=[(10.0, 1.0, 0.0)],
        weight_items=weights,
        current_ids={"A", "D"}, target_ids={"A"},
        installed_mass_limit=20.0, residual_moment_limit=10.0,
    )
    assert plan.feasible is True
    assert [(s.action, s.weight_id) for s in plan.steps] == [
        ("remove", "D")
    ]
    for s in plan.steps:
        assert s.total_mass <= 20.0 + EPS
        assert s.residual_magnitude <= 10.0 + EPS


# ---------------------------------------------------------------------------
# 随机对拍：与独立暴力 BFS 比较长度、择优键、可达数与阻断动作
# ---------------------------------------------------------------------------


def test_random_cross_check_against_reference_bfs():
    rng = random.Random(20260930)
    for trial in range(300):
        n = rng.randint(1, 5)
        weights = []
        for k in range(n):
            weights.append((
                f"w{k}",
                rng.choice([0.0, 0.5, 1.0, 2.0, 3.0]),
                rng.choice([0.0, 0.5, 1.0, 2.0]),
                rng.choice([0.0, 30.0, 90.0, 150.0, 180.0, 210.0, 270.0]),
            ))
        fixed = [
            (rng.choice([0.0, 1.0, 2.0, 4.0]),
             rng.choice([0.0, 1.0]),
             rng.choice([0.0, 90.0, 180.0]))
            for _ in range(rng.randint(0, 2))
        ]
        ids = [w[0] for w in weights]
        cur = {w for w in ids if rng.random() < 0.4}
        tgt = {w for w in ids if rng.random() < 0.4}
        mass_lim = rng.choice([4.0, 6.0, 10.0, 100.0])
        mom_lim = rng.choice([0.0, 2.0, 5.0, 100.0])

        ref = reference_plan(
            fixed, weights, cur, tgt, mass_lim, mom_lim
        )
        if ref["feasible"] is None:
            # 起止越限：产品必须抛 422
            with pytest.raises(WeightSequenceValidationError):
                plan_weight_sequence(
                    fixed_parts=fixed, weight_items=weights,
                    current_ids=cur, target_ids=tgt,
                    installed_mass_limit=mass_lim,
                    residual_moment_limit=mom_lim,
                )
            continue

        plan = plan_weight_sequence(
            fixed_parts=fixed, weight_items=weights,
            current_ids=cur, target_ids=tgt,
            installed_mass_limit=mass_lim,
            residual_moment_limit=mom_lim,
        )

        assert plan.reachable_states == ref["reachable"], trial
        assert plan.feasible == ref["feasible"], trial
        if ref["feasible"]:
            got = [(s.action, s.weight_id) for s in plan.steps]
            assert len(got) == ref["length"], trial
            assert got == list(ref["actions"]), trial
            assert plan.max_intermediate_magnitude == pytest.approx(
                ref["max_mag"], abs=1e-9
            ), trial
            # 终态恰为目标（临时件已全部拆除）
            assert set(plan.steps[-1].installed_ids) == tgt if plan.steps \
                else set(plan.initial.installed_ids) == tgt
            # 所有中间状态合法
            for s in plan.steps:
                assert s.total_mass <= mass_lim + EPS
                assert s.residual_magnitude <= mom_lim + EPS
        else:
            b = plan.nearest_blocked_action
            assert b is not None
            assert (
                b.moment_excess,
                b.mass_excess,
                (b.action, b.weight_id),
                b.from_depth,
            ) == (
                pytest.approx(ref["blocked_key"][0], abs=1e-9),
                pytest.approx(ref["blocked_key"][1], abs=1e-9),
                (ref["blocked_key"][2], ref["blocked_key"][3]),
                ref["blocked_key"][4],
            ), trial


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def _payload(**over):
    body = {
        "fixed_parts": [{"mass": 10.0, "radius": 1.0, "azimuth": 0.0}],
        "weights": [
            {"id": "A", "mass": 5.0, "radius": 2.0, "azimuth": 180.0},
            {"id": "B", "mass": 5.0, "radius": 2.0, "azimuth": 150.0},
            {"id": "C", "mass": 5.0, "radius": 2.0, "azimuth": 210.0},
            {"id": "T", "mass": 2.0, "radius": 4.0, "azimuth": 180.0},
        ],
        "current_ids": ["A"],
        "target_ids": ["B", "C"],
        "installed_mass_limit": 20.0,
        "residual_moment_limit": 9.0,
    }
    body.update(over)
    return body


def test_api_feasible_temporary_bridge():
    r = client.post("/api/v1/plan-weight-sequence", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert len(body["steps"]) == 5
    assert body["temporary_weight_ids"] == ["T"]
    assert body["reachable_states"] == 8
    assert body["details"]["explored_states"] == 16
    assert body["initial_state"]["installed_ids"] == ["A"]
    assert body["target_state"]["installed_ids"] == ["B", "C"]
    seq = [(s["action"], s["weight_id"]) for s in body["steps"]]
    assert seq[0] == ("install", "T")
    assert seq[-1] == ("install", "C")
    assert body["steps"][-1]["installed_ids"] == ["B", "C"]
    # 每步总质量/残余不越限，模长与分量自洽
    for s in body["steps"]:
        assert s["total_mass"] <= 20.0 + 1e-9
        assert s["residual_magnitude"] <= 9.0 + 1e-9
        assert s["residual_magnitude"] == pytest.approx(
            math.hypot(s["residual"]["east"], s["residual"]["north"]),
            abs=1e-9,
        )


def test_api_infeasible_blocked_action():
    body = _payload(
        weights=[
            {"id": "A", "mass": 5.0, "radius": 2.0, "azimuth": 180.0},
            {"id": "B", "mass": 2.5, "radius": 2.0, "azimuth": 170.0},
            {"id": "C", "mass": 2.5, "radius": 2.0, "azimuth": 190.0},
        ],
        residual_moment_limit=5.0,
    )
    r = client.post("/api/v1/plan-weight-sequence", json=body)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is False
    assert b["steps"] == []
    assert b["reachable_states"] == 3
    blk = b["nearest_blocked_action"]
    assert blk["action"] == "remove"
    assert blk["weight_id"] == "A"
    assert blk["from_depth"] == 1
    assert blk["moment_excess"] == pytest.approx(0.149682, abs=1e-6)
    assert blk["mass_excess"] == 0.0


def test_api_422_current_violates():
    body = _payload(
        weights=[{"id": "A", "mass": 1.0, "radius": 0.0, "azimuth": 0.0}],
        target_ids=[],
        residual_moment_limit=5.0,
    )
    r = client.post("/api/v1/plan-weight-sequence", json=body)
    assert r.status_code == 422
    p = r.json()
    assert p["reason"] == "current_state_violates_limits"
    assert p["endpoint"] == "current"
    assert p["moment_excess"] == pytest.approx(5.0, abs=1e-9)


def test_api_422_target_violates():
    # 起点 {C} 抵消固定矩（合法），目标零矩 {A} 残余 10 越限
    body = _payload(
        weights=[
            {"id": "C", "mass": 10.0, "radius": 1.0, "azimuth": 180.0},
            {"id": "A", "mass": 1.0, "radius": 0.0, "azimuth": 0.0},
        ],
        current_ids=["C"],
        target_ids=["A"],
        residual_moment_limit=5.0,
    )
    r = client.post("/api/v1/plan-weight-sequence", json=body)
    assert r.status_code == 422
    assert r.json()["reason"] == "target_state_violates_limits"


def test_api_duplicate_current_ids_dedup_ok():
    body = _payload(current_ids=["A", "A"], target_ids=["B", "B", "C", "C"])
    r = client.post("/api/v1/plan-weight-sequence", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["feasible"] is True


def test_api_validation_errors():
    # 超过 12 件配重
    too_many = _payload(weights=[
        {"id": f"w{i:02d}", "mass": 1.0, "radius": 1.0, "azimuth": 0.0}
        for i in range(13)
    ], current_ids=[], target_ids=[])
    assert client.post(
        "/api/v1/plan-weight-sequence", json=too_many
    ).status_code == 422

    # ID 重复
    dup = _payload(weights=body_weights_duplicate())
    assert client.post(
        "/api/v1/plan-weight-sequence", json=dup
    ).status_code == 422

    # 未知 ID
    assert client.post(
        "/api/v1/plan-weight-sequence", json=_payload(current_ids=["A", "Z"])
    ).status_code == 422
    assert client.post(
        "/api/v1/plan-weight-sequence", json=_payload(target_ids=["ZZ"])
    ).status_code == 422

    # 空 ID
    bad = _payload(weights=[
        {"id": "", "mass": 1.0, "radius": 1.0, "azimuth": 0.0}
    ])
    assert client.post(
        "/api/v1/plan-weight-sequence", json=bad
    ).status_code == 422

    # 负质量 / 负半径 / 负限值
    bad = _payload(weights=[
        {"id": "A", "mass": -1.0, "radius": 1.0, "azimuth": 0.0}
    ])
    assert client.post(
        "/api/v1/plan-weight-sequence", json=bad
    ).status_code == 422
    assert client.post(
        "/api/v1/plan-weight-sequence",
        json=_payload(installed_mass_limit=-0.1),
    ).status_code == 422
    assert client.post(
        "/api/v1/plan-weight-sequence",
        json=_payload(residual_moment_limit=-0.1),
    ).status_code == 422

    # 固定部件负质量
    bad = _payload(fixed_parts=[{"mass": -1.0, "radius": 1.0, "azimuth": 0.0}])
    assert client.post(
        "/api/v1/plan-weight-sequence", json=bad
    ).status_code == 422


def body_weights_duplicate():
    return [
        {"id": "A", "mass": 1.0, "radius": 1.0, "azimuth": 0.0},
        {"id": "A", "mass": 2.0, "radius": 1.0, "azimuth": 90.0},
        {"id": "B", "mass": 1.0, "radius": 1.0, "azimuth": 0.0},
        {"id": "T", "mass": 2.0, "radius": 4.0, "azimuth": 180.0},
    ]


def test_api_rejects_non_finite_at_model_layer():
    from pydantic import ValidationError

    from app.models import PlanWeightSequenceRequest

    valid = {
        "fixed_parts": [],
        "weights": [
            {"id": "A", "mass": 1.0, "radius": 1.0, "azimuth": 0.0}
        ],
        "current_ids": [],
        "target_ids": ["A"],
        "installed_mass_limit": 10.0,
        "residual_moment_limit": 1.0,
    }
    for path, value in [
        (("installed_mass_limit",), float("nan")),
        (("residual_moment_limit",), float("inf")),
        (("weights", 0, "mass"), float("nan")),
        (("weights", 0, "azimuth"), float("inf")),
    ]:
        body = {**valid}
        target = body
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValidationError):
            PlanWeightSequenceRequest.model_validate(body)
