"""圆顶检修配重换装顺序规划：安全子集 BFS 与 HTTP API 测试。"""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner.swap import (
    ACTION_TYPE_NAMES,
    INSTALL,
    REMOVE,
    SwapValidationError,
    Weight,
    plan_weight_swap,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _weight(wid, mass, radius, az):
    a = math.radians(az)
    return Weight(
        id=wid, mass=mass, radius=radius, azimuth_deg=az,
        mx=mass * radius * math.sin(a),
        my=mass * radius * math.cos(a),
    )


def _payload(fixed, weights, current, target, mass_limit, moment_limit):
    return {
        "fixed_parts": [
            {"mass": m, "radius": r, "azimuth": a} for (m, r, a) in fixed
        ],
        "weights": [
            {"id": i, "mass": m, "radius": r, "azimuth": a}
            for (i, m, r, a) in weights
        ],
        "current_ids": list(current),
        "target_ids": list(target),
        "installed_mass_limit": mass_limit,
        "residual_moment_limit": moment_limit,
    }


def _actions(plan):
    return [(ACTION_TYPE_NAMES[s.action_type], s.weight_id) for s in plan.steps]


def _mags(plan):
    return [s.residual_magnitude for s in plan.steps]


# ---------------------------------------------------------------------------
# 基础：零步、直连、固定部件矩约定
# ---------------------------------------------------------------------------


def test_zero_steps_when_current_equals_target():
    ws = [_weight("A", 10, 1, 180)]
    plan = plan_weight_swap([], ws, ["A"], ["A"], 100.0, 100.0)
    assert plan.feasible is True
    assert plan.steps == ()
    assert plan.reachable_states >= 1


def test_moment_convention_north_zero_east_positive():
    # 固定 10@0° → (east=0, north=+10)
    plan = plan_weight_swap(
        [(10.0, 1.0, 0.0)], [], [], [], 100.0, 100.0
    )
    assert plan.start_x == pytest.approx(0.0, abs=1e-12)
    assert plan.start_y == pytest.approx(10.0, abs=1e-12)
    # 90° 配重贡献东向
    plan2 = plan_weight_swap(
        [], [_weight("E", 3.0, 2.0, 90.0)], ["E"], ["E"], 100.0, 100.0
    )
    assert plan2.start_x == pytest.approx(6.0, abs=1e-12)
    assert plan2.start_y == pytest.approx(0.0, abs=1e-12)


def test_install_before_remove_when_tied():
    # A/B 均为 10kg@180°，固定 10@0° 平衡；两种 2 步路径中间峰值都是 10，
    # 字典序 install < remove → 先装 B 后拆 A
    ws = [_weight("A", 10, 1, 180), _weight("B", 10, 1, 180)]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 30.0, 10.0)
    assert plan.feasible is True
    assert _actions(plan) == [("install", "B"), ("remove", "A")]
    assert plan.steps[1].installed_ids == ("B",)
    assert plan.steps[1].residual_magnitude == pytest.approx(0.0, abs=1e-9)


def test_remove_first_when_intermediate_residual_smaller():
    # A=11kg@180（残余 -1y），B=10kg@180：
    # 先装路径经 {A,B}（残余 -11y，峰值 11）；先拆路径经空集（+10y，峰值 10）。
    # 最大中间残余优先于动作字典序 → 先拆 A。
    ws = [_weight("A", 11, 1, 180), _weight("B", 10, 1, 180)]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 30.0, 11.0)
    assert plan.feasible is True
    assert _actions(plan) == [("remove", "A"), ("install", "B")]
    assert max(_mags(plan)) == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 临时配重
# ---------------------------------------------------------------------------


def test_temporary_weight_required_bridge_and_removed_at_end():
    # 固定 +10y；A/B=10kg@180 r=1；T=2.5kg@180 r=2（矩 -5y）。
    # 质量上限 25：先装 B 需 30kg 被总质量阻断；先拆 A 残余 10 被力矩（限 5）
    # 阻断 → 必须借 T 桥接 4 步，且终点 T 已拆除。
    ws = [
        _weight("A", 10, 1, 180),
        _weight("B", 10, 1, 180),
        _weight("T", 2.5, 2, 180),
    ]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 25.0, 5.0)
    assert plan.feasible is True
    assert _actions(plan) == [
        ("install", "T"),
        ("remove", "A"),
        ("install", "B"),
        ("remove", "T"),
    ]
    # 每步结束都不越限
    for s in plan.steps:
        assert s.total_mass <= 25.0 + 1e-9
        assert s.residual_magnitude <= 5.0 + 1e-9
    # 临时配重最终拆除，终点恰为目标集
    assert plan.steps[-1].installed_ids == ("B",)


def test_without_temporary_weight_no_path_and_nearest_blocked():
    ws = [_weight("A", 10, 1, 180), _weight("B", 10, 1, 180)]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 25.0, 5.0)
    assert plan.feasible is False
    assert plan.steps == ()
    # 起点是唯一可达状态
    assert plan.reachable_states == 1
    b = plan.nearest_blocked
    assert b is not None
    # 两个阻断动作：install B（质量超限 5 且力矩超限 5）、remove A（仅力矩超限 5）；
    # 质量矩超限量并列时取总质量超限量小者 → remove A
    assert ACTION_TYPE_NAMES[b.action_type] == "remove"
    assert b.weight_id == "A"
    assert b.moment_excess == pytest.approx(5.0, abs=1e-9)
    assert b.mass_excess == pytest.approx(0.0, abs=1e-12)
    assert b.from_installed_ids == ("A",)


def test_nearest_blocked_prefers_smaller_moment_excess():
    # 固定 +10y，A=10@180 在位平衡；质量上限 30，力矩限值 8。
    # remove A：落点矩 10 → moment_excess 2；install B 落点矩 10 → 同样 2，
    # 但 install B 总质量 30 不超限 → remove A（mass_excess=0）与 install B
    # （mass_excess=0）并列 → 动作字典序 install 更小
    ws = [_weight("A", 10, 1, 180), _weight("B", 10, 1, 180)]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 30.0, 8.0)
    assert plan.feasible is False
    b = plan.nearest_blocked
    assert (ACTION_TYPE_NAMES[b.action_type], b.weight_id) == ("install", "B")
    assert b.moment_excess == pytest.approx(2.0, abs=1e-9)


def test_mass_excess_breaks_moment_tie():
    # 让两个阻断动作的力矩超限量相同：install 落点总质量超限更多。
    ws = [_weight("A", 10, 1, 180), _weight("B", 10, 1, 180)]
    # 质量上限 25：install B 落点 30 → mass_excess 5；remove A mass_excess 0。
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["B"], 25.0, 8.0)
    assert plan.feasible is False
    b = plan.nearest_blocked
    assert (ACTION_TYPE_NAMES[b.action_type], b.weight_id) == ("remove", "A")
    assert b.moment_excess == pytest.approx(2.0, abs=1e-9)
    assert b.mass_excess == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 起点 / 终点越限 → 422
# ---------------------------------------------------------------------------


def test_start_violation_raises():
    ws = [_weight("A", 10, 1, 180)]
    # 空起点：固定矩 10 > 限值 2
    with pytest.raises(SwapValidationError) as exc:
        plan_weight_swap([(10, 1, 0)], ws, [], ["A"], 30.0, 2.0)
    assert exc.value.payload["endpoint"] == "start"
    assert exc.value.payload["moment_excess"] == pytest.approx(8.0, abs=1e-9)


def test_goal_mass_violation_raises():
    ws = [_weight("A", 20, 0, 0)]
    # 终点 {A}：固定 10 + 20 = 30 > 质量上限 25（A 半径 0 不贡献矩）
    with pytest.raises(SwapValidationError) as exc:
        plan_weight_swap([(10, 0, 0)], ws, [], ["A"], 25.0, 100.0)
    assert exc.value.payload["endpoint"] == "goal"
    assert exc.value.payload["mass_excess"] == pytest.approx(5.0, abs=1e-9)


def test_boundary_is_compliant():
    # 恰在限值边界（浮点精确）视为安全
    ws = [_weight("A", 10, 1, 180)]
    plan = plan_weight_swap([(10, 1, 0)], ws, ["A"], ["A"], 20.0, 0.0)
    assert plan.feasible is True


# ---------------------------------------------------------------------------
# 穷举对拍：与朴素“枚举全部安全路径”结果一致
# ---------------------------------------------------------------------------


def _brute_force_optimal(fixed, weights, current, target, mlim, rlim):
    """朴素 BFS（逐掩码，按动作字典序扩展）+ 同层最优键保留，作参照实现。"""
    ids = tuple(sorted(w.id for w in weights))
    by_id = {w.id: w for w in weights}
    n = len(ids)
    idx = {wid: i for i, wid in enumerate(ids)}

    def vec_of(wid):
        w = by_id[wid]
        a = math.radians(w.azimuth_deg)
        return (w.mass * w.radius * math.sin(a),
                w.mass * w.radius * math.cos(a), w.mass)

    fx = sum(m * r * math.sin(math.radians(a)) for (m, r, a) in fixed)
    fy = sum(m * r * math.cos(math.radians(a)) for (m, r, a) in fixed)
    fm = sum(m for (m, r, a) in fixed)
    cache = {}

    def measure(mask):
        if mask in cache:
            return cache[mask]
        x, y, mm = fx, fy, fm
        for i in range(n):
            if mask & (1 << i):
                vx, vy, mass = vec_of(ids[i])
                x += vx
                y += vy
                mm += mass
        res = (mm, x, y, math.hypot(x, y))
        cache[mask] = res
        return res

    def safe(mask):
        mm, _x, _y, mag = measure(mask)
        return mm <= mlim + 1e-9 and mag <= rlim + 1e-9

    start = sum(1 << idx[i] for i in current)
    goal = sum(1 << idx[i] for i in target)

    # dist + 到每个掩码的最优（步数由层保证；层内比较键）
    best = {start: ((), 0.0)}
    frontier = [start]
    reachable = 1
    while frontier:
        layer_cands = {}
        for mask in frontier:
            acts, peak = best[mask]
            for i in range(n):  # install 先（ID 升序）
                bit = 1 << i
                if mask & bit:
                    continue
                child = mask | bit
                if child in best:
                    continue
                if not safe(child):
                    continue
                mag = measure(child)[3]
                key = (acts + (("install", ids[i]),), max(peak, mag))
                if child not in layer_cands or _brute_key_less(key, layer_cands[child]):
                    layer_cands[child] = key
            for i in range(n):  # remove
                bit = 1 << i
                if not mask & bit:
                    continue
                child = mask ^ bit
                if child in best:
                    continue
                if not safe(child):
                    continue
                mag = measure(child)[3]
                key = (acts + (("remove", ids[i]),), max(peak, mag))
                if child not in layer_cands or _brute_key_less(key, layer_cands[child]):
                    layer_cands[child] = key
        for child, key in layer_cands.items():
            best[child] = key
            reachable += 1
        frontier = list(layer_cands)

    if goal not in best:
        return None, reachable
    return best[goal][0], reachable


def _brute_key_less(a, b):
    pa, ma = a
    pb, mb = b
    if abs(ma - mb) > 1e-9:
        return ma < mb
    if abs(ma) <= 1e-9 and abs(mb) <= 1e-9:
        pass
    return pa < pb


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force(seed):
    import random

    rng = random.Random(seed)
    n = rng.randint(1, 6)
    ids = [f"w{i}" for i in range(n)]
    weights = [
        _weight(
            ids[i],
            rng.choice([0.0, 1.0, 2.0, 3.0, 5.0]),
            rng.choice([0.0, 1.0, 2.0]),
            rng.choice([0.0, 60.0, 90.0, 120.0, 180.0, 240.0, 270.0]),
        )
        for i in range(n)
    ]
    fixed = [(rng.choice([0.0, 4.0, 6.0, 10.0]), 1.0,
              rng.choice([0.0, 90.0, 180.0]))]
    cur = {i for i in ids if rng.random() < 0.4}
    tgt = {i for i in ids if rng.random() < 0.4}
    mlim = rng.choice([10.0, 15.0, 25.0, 60.0])
    rlim = rng.choice([2.0, 5.0, 8.0, 20.0])

    def measure_idset(idset):
        m = sum(p[0] for p in fixed)
        x = sum(p[0] * p[1] * math.sin(math.radians(p[2])) for p in fixed)
        y = sum(p[0] * p[1] * math.cos(math.radians(p[2])) for p in fixed)
        for wid in idset:
            w = next(z for z in weights if z.id == wid)
            m += w.mass
            x += w.mx
            y += w.my
        return m, math.hypot(x, y)

    cm, cmag = measure_idset(cur)
    gm, gmag = measure_idset(tgt)
    if cm > mlim + 1e-9 or cmag > rlim + 1e-9:
        pytest.skip("起点越限")
    if gm > mlim + 1e-9 or gmag > rlim + 1e-9:
        pytest.skip("终点越限")

    plan = plan_weight_swap(fixed, weights, sorted(cur), sorted(tgt), mlim, rlim)
    ref, reachable = _brute_force_optimal(
        fixed, weights, sorted(cur), sorted(tgt), mlim, rlim
    )
    assert plan.reachable_states == reachable
    if ref is None:
        assert plan.feasible is False
        assert plan.steps == ()
    else:
        assert plan.feasible is True
        assert _actions(plan) == list(ref)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_api_feasible_example():
    r = client.post(
        "/api/v1/plan-weight-swap",
        json=_payload(
            fixed=[(10, 1, 0)],
            weights=[
                ("A", 10, 1, 180),
                ("B", 10, 1, 180),
                ("T", 2.5, 2, 180),
            ],
            current=["A"],
            target=["B"],
            mass_limit=25.0,
            moment_limit=5.0,
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert body["reachable_states"] == 5
    assert [(s["action"], s["weight_id"]) for s in body["steps"]] == [
        ("install", "T"),
        ("remove", "A"),
        ("install", "B"),
        ("remove", "T"),
    ]
    last = body["steps"][-1]
    assert last["installed_ids"] == ["B"]
    assert last["residual_magnitude"] == 0.0
    assert body["nearest_blocked"] is None
    # 每步安全
    for s in body["steps"]:
        assert s["total_mass"] <= 25.0 + 1e-9
        assert s["residual_magnitude"] <= 5.0 + 1e-9


def test_api_no_path():
    r = client.post(
        "/api/v1/plan-weight-swap",
        json=_payload(
            fixed=[(10, 1, 0)],
            weights=[("A", 10, 1, 180), ("B", 10, 1, 180)],
            current=["A"],
            target=["B"],
            mass_limit=25.0,
            moment_limit=5.0,
        ),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["steps"] == []
    assert body["reachable_states"] == 1
    nb = body["nearest_blocked"]
    assert (nb["action"], nb["weight_id"]) == ("remove", "A")
    assert nb["moment_excess"] == 5.0
    assert nb["mass_excess"] == 0.0


def test_api_start_violation_422():
    r = client.post(
        "/api/v1/plan-weight-swap",
        json=_payload(
            fixed=[(10, 1, 0)],
            weights=[("A", 10, 1, 180)],
            current=[],
            target=["A"],
            mass_limit=30.0,
            moment_limit=2.0,
        ),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["endpoint"] == "start"
    assert body["moment_excess"] == 8.0
    assert body["installed_ids"] == []


def test_api_goal_violation_422():
    r = client.post(
        "/api/v1/plan-weight-swap",
        json=_payload(
            fixed=[(10, 0, 0)],
            weights=[("A", 20, 0, 0)],
            current=[],
            target=["A"],
            mass_limit=25.0,
            moment_limit=100.0,
        ),
    )
    assert r.status_code == 422
    assert r.json()["endpoint"] == "goal"


def test_api_validation_errors():
    base = _payload(
        fixed=[],
        weights=[("A", 1, 1, 0), ("B", 1, 1, 0)],
        current=["A"],
        target=["B"],
        mass_limit=10.0,
        moment_limit=10.0,
    )

    # 超过 12 件
    too_many = dict(base)
    too_many["weights"] = [
        {"id": f"W{i}", "mass": 1, "radius": 1, "azimuth": 0}
        for i in range(13)
    ]
    too_many["current_ids"] = []
    too_many["target_ids"] = []
    assert client.post("/api/v1/plan-weight-swap", json=too_many).status_code == 422

    # 重复 ID
    dup = dict(base)
    dup["weights"] = [
        {"id": "A", "mass": 1, "radius": 1, "azimuth": 0},
        {"id": "A", "mass": 1, "radius": 1, "azimuth": 0},
    ]
    assert client.post("/api/v1/plan-weight-swap", json=dup).status_code == 422

    # 引用未知 ID
    unknown = dict(base)
    unknown["current_ids"] = ["ZZZ"]
    assert client.post("/api/v1/plan-weight-swap", json=unknown).status_code == 422

    # 重复 current_ids
    rep = dict(base)
    rep["current_ids"] = ["A", "A"]
    assert client.post("/api/v1/plan-weight-swap", json=rep).status_code == 422

    # 负质量 / NaN
    bad = dict(base)
    bad["weights"] = [{"id": "A", "mass": -1, "radius": 1, "azimuth": 0}]
    assert client.post("/api/v1/plan-weight-swap", json=bad).status_code == 422

    # 空 weights
    empty = dict(base)
    empty["weights"] = []
    empty["current_ids"] = []
    empty["target_ids"] = []
    assert client.post("/api/v1/plan-weight-swap", json=empty).status_code == 422


def test_api_request_array_order_does_not_matter():
    def post(weight_order, cur_order):
        body = _payload(
            fixed=[(10, 1, 0)],
            weights=weight_order,
            current=cur_order,
            target=["B"],
            mass_limit=30.0,
            moment_limit=10.0,
        )
        return client.post("/api/v1/plan-weight-swap", json=body).json()

    w_sets = [
        [("B", 10, 1, 180), ("A", 10, 1, 180)],
        [("A", 10, 1, 180), ("B", 10, 1, 180)],
    ]
    b1 = post(w_sets[0], ["A"])
    b2 = post(w_sets[1], ["A"])
    assert (
        [(s["action"], s["weight_id"]) for s in b1["steps"]]
        == [(s["action"], s["weight_id"]) for s in b2["steps"]]
    )
    # 在位 ID 始终按 ID 升序输出
    assert all(
        s["installed_ids"] == sorted(s["installed_ids"])
        for s in b1["steps"]
    )
