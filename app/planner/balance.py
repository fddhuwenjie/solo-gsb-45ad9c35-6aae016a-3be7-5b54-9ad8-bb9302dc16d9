"""圆顶静平衡配重规划：候选槽位枚举 + 偏差情景穷举。

静平衡以绕旋转轴的**质量矩向量**（二维，方位沿用北为 0°、向东递增）描述：

- 已装部件 ``j``（质量 ``m_j``、半径 ``r_j``、方位 ``a_j``）贡献
  ``M0 = Σ m_j·r_j·(sin a_j, cos a_j)``；
- 槽位 ``s`` 上放置标称质量 ``m`` 的配重后贡献
  ``C = m·r_s·(sin a_s, cos a_s)``；
- 残余质量矩 ``R = M0 + C``（配重与已装部件同向叠加，平衡时配重应加在
  原始质量矩的**反方向**）。

枚举
----

每个槽位的可选标称质量为 ``0, mass_step, 2·mass_step, …`` 且不超过该槽位
``mass_limit``，非零档位最多 40 个（由请求模型保证）。方案最多使用两个
槽位，因此对“零配重、单槽、两槽”三类方案做**精确完全枚举**。

偏差情景
--------

每个非零配重的实际质量可取
``标称·(1 - tol_m)``、``标称``、``标称·(1 + tol_m)``
（``tol_m = mass_tolerance_ratio``），实际方位可取
``标称 - tol_a``、``标称``、``标称 + tol_a``
（``tol_a = azimuth_tolerance`` 度）。对每个方案的 9^k 个质量/方位组合
（k 为非零配重数，两槽时最多 81 个）穷举残余向量，取模长最大者为
**最坏残余质量矩**；情景并列时取偏差符号按配重顺序字典序最小者
（枚举顺序即符号顺序 -1 < 0 < +1，首情景采纳后仅在模长显著更大时
替换）。

择优
----

候选方案依次按以下键比较，差值不超过 ``TIE_EPS = 1e-9`` 视为并列：

1. 最坏残余质量矩模长（小者优）；
2. 标称残余质量矩模长（小者优）；
3. 总配重质量（小者优）；
4. 按槽位 ID 排列的质量向量（字典序小者优；未使用槽位质量记 0）。

枚举顺序本身确定（单槽按槽位顺序、质量升序；两槽按槽位对与两槽质量
字典序升序），保证完全并列时结果可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 择优键并列容差：任何标量键差值不超过它即视为并列
TIE_EPS = 1e-9

# 以下上限由请求模型（app.models）强制：槽位数、每槽非零档位数、方案
# 最多使用槽位数；规划器本身按“至多两个配重”展开偏差情景。
MAX_SLOTS = 8
MAX_LEVELS_PER_SLOT = 40
MAX_USED_SLOTS = 2

# 偏差符号（相对标称）：负偏差、标称、正偏差；顺序即最坏情景并列时的
# 字典序（-1 < 0 < +1），也是情景穷举顺序。
_SIGNS = (-1, 0, 1)

# 单个配重的 9 个情景（质量符号, 方位符号）扁平代码，质量符号在外：
# (-1,-1),(-1,0),(-1,+1),(0,-1),…,(+1,+1)。
_SCENARIO_CODES = tuple((sm, sa) for sm in _SIGNS for sa in _SIGNS)


@dataclass(frozen=True)
class Slot:
    """候选配重槽位。"""

    id: str
    radius: float
    azimuth_deg: float
    mass_limit: float
    # 预计算的非零标称质量档位（升序）
    levels: tuple[float, ...]


@dataclass(frozen=True)
class ChosenWeight:
    """方案中某个非零配重（标称值 + 最坏情景的实际取值）。"""

    slot_id: str
    radius: float
    nominal_mass: float
    nominal_azimuth_deg: float
    worst_mass: float
    worst_azimuth_deg: float


@dataclass(frozen=True)
class WeightScenario:
    """单个配重在最坏验收情景中的实际取值。"""

    slot_id: str
    mass: float
    azimuth_deg: float
    mass_sign: int  # -1 / 0 / +1
    azimuth_sign: int


@dataclass(frozen=True)
class BalancePlan:
    """最终择优结果。"""

    original_x: float
    original_y: float
    original_magnitude: float
    weights: tuple[ChosenWeight, ...]
    total_balance_mass: float
    nominal_x: float
    nominal_y: float
    nominal_magnitude: float
    worst_x: float
    worst_y: float
    worst_magnitude: float
    # 最坏偏差组合，顺序与 weights（按槽位 ID 排序）一致
    worst_combo: tuple[WeightScenario, ...]
    residual_limit: float
    compliant: bool
    evaluated_plans: int
    max_scenarios_per_plan: int
    # 按槽位 ID 排列的质量向量（含未使用槽位的 0）
    mass_vector: tuple[float, ...]
    slot_ids: tuple[str, ...] = field(default_factory=tuple)


def _unit(azimuth_deg: float) -> tuple[float, float]:
    """方位角（度，北 0° 向东递增）→ 平面单位向量 (x=东向, y=北向)。"""

    a = math.radians(azimuth_deg)
    return math.sin(a), math.cos(a)


def _vec(mass: float, radius: float, azimuth_deg: float) -> tuple[float, float]:
    ux, uy = _unit(azimuth_deg)
    return mass * radius * ux, mass * radius * uy


def slot_levels(mass_limit: float, mass_step: float) -> tuple[float, ...]:
    """槽位的非零标称质量档位 ``step, 2·step, …``（不超过 mass_limit）。

    末档恰在浮点边界上时用 ``1e-9`` 相对容差纳入，避免 ``n·step`` 与
    ``mass_limit`` 相差一个舍入误差时丢档。
    """

    n = int(math.floor(mass_limit / mass_step + 1e-9))
    return tuple(k * mass_step for k in range(1, n + 1))


def _strictly_less_scalar(a: float, b: float) -> bool:
    """``a`` 是否严格小于 ``b``（差值不超过 TIE_EPS 视为并列）。

    特别地，两个都贴近零（绝对值不超过 TIE_EPS，含浮点残差如 1e-15）
    的值也视为并列，避免“精确抵消”方案被噪声判出优劣。
    """

    if abs(a) <= TIE_EPS and abs(b) <= TIE_EPS:
        return False
    return a < b and b - a > TIE_EPS


def _strictly_less_vector(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """并列感知的质量向量字典序：``a`` 是否严格小于 ``b``。

    逐分量比较：首个差值绝对值超过 TIE_EPS 的分量决定序关系；
    全部分量并列则视为相等（非严格小于）。
    """

    for x, y in zip(a, b):
        if abs(x - y) <= TIE_EPS:
            continue
        return x < y
    return False


@dataclass
class _PrecomputedSlot:
    """枚举期间一个槽位的预计算量。"""

    slot: Slot
    # 每档标称质量的 (质量, 标称矩 x, 标称矩 y)
    nominal: list[tuple[float, float, float]]
    # scenario[k][flat] = 档位 k 在偏差情景 flat 下的实际质量矩；
    # flat = sm_index*3 + sa_index（sm/sa_index ∈ {0,1,2} ↔ 符号 -1,0,+1）
    scenario: list[list[tuple[float, float]]]
    # 与 scenario 平行的实际质量 / 实际方位矩阵（索引 [k][sm_index][sa_index]）
    actual_mass: list[list[list[float]]]
    actual_azimuth: list[list[list[float]]]


def _precompute(slot: Slot, mass_tol: float, az_tol_deg: float) -> _PrecomputedSlot:
    nominal: list[tuple[float, float, float]] = []
    scenario: list[list[tuple[float, float]]] = []
    actual_mass: list[list[list[float]]] = []
    actual_az: list[list[list[float]]] = []
    for m in slot.levels:
        ux0, uy0 = _unit(slot.azimuth_deg)
        nominal.append((m, m * slot.radius * ux0, m * slot.radius * uy0))
        per_level: list[tuple[float, float]] = []
        mm_grid: list[list[float]] = []
        az_grid: list[list[float]] = []
        for sm in _SIGNS:
            mp = m * (1.0 + sm * mass_tol)
            per_sm: list[tuple[float, float]] = []
            row_m: list[float] = []
            row_az: list[float] = []
            for sa in _SIGNS:
                ap = slot.azimuth_deg + sa * az_tol_deg
                per_sm.append(_vec(mp, slot.radius, ap))
                row_m.append(mp)
                row_az.append(ap)
            per_level.extend(per_sm)
            mm_grid.append(row_m)
            az_grid.append(row_az)
        scenario.append(per_level)
        actual_mass.append(mm_grid)
        actual_az.append(az_grid)
    return _PrecomputedSlot(slot, nominal, scenario, actual_mass, actual_az)


@dataclass(frozen=True)
class _Entry:
    """单个非零配重在方案中的位置（槽位下标 + 档位下标）。"""

    slot_index: int
    level_index: int


# (最坏模长, 标称模长, 总质量, 质量向量)
_Key = tuple[float, float, float, tuple[float, ...]]


def _strictly_better(candidate: _Key, incumbent: _Key) -> bool:
    """并列感知字典序：candidate 是否严格优于 incumbent。"""

    if _strictly_less_scalar(candidate[0], incumbent[0]):
        return True
    if _strictly_less_scalar(incumbent[0], candidate[0]):
        return False
    if _strictly_less_scalar(candidate[1], incumbent[1]):
        return True
    if _strictly_less_scalar(incumbent[1], candidate[1]):
        return False
    if _strictly_less_scalar(candidate[2], incumbent[2]):
        return True
    if _strictly_less_scalar(incumbent[2], candidate[2]):
        return False
    return _strictly_less_vector(candidate[3], incumbent[3])


def plan_balance(
    parts: list[tuple[float, float, float]],
    slots: list[Slot],
    mass_step: float,
    mass_tolerance_ratio: float,
    azimuth_tolerance_deg: float,
    residual_moment_limit: float,
) -> BalancePlan:
    """完全枚举所有 ≤2 槽方案并按择优键返回最优方案。

    参数
    ----
    parts:
        已装部件 ``(质量, 旋转轴半径, 方位角)`` 三元组列表。
    slots:
        候选槽位（调用方需保证 ID 去重、非零档位数不超过上限）。

    即使没有任何方案满足残余限值，也返回按择优键最优的方案，
    由 ``compliant`` 标明是否达标。
    """

    original_x = original_y = 0.0
    for mass, radius, azimuth in parts:
        vx, vy = _vec(mass, radius, azimuth)
        original_x += vx
        original_y += vy
    original_mag = math.hypot(original_x, original_y)

    # 规划器内部一律按槽位 ID 升序工作：枚举顺序、择优键中的质量向量与
    # 输出顺序都以 ID 顺序为准，结果与请求数组顺序无关。
    slots = sorted(slots, key=lambda s: s.id)
    prepared = [
        _precompute(s, mass_tolerance_ratio, azimuth_tolerance_deg) for s in slots
    ]
    n_slots = len(slots)

    # 每个 (槽位, 档位) 的 3×3 偏差情景质量矩，预展平为 9 元组，
    # 使方案枚举内层只有元组索引与向量加法。
    scen_flat: list[list[tuple[tuple[float, float], ...]]] = [
        [tuple(tuple(x) for x in pc.scenario[li])
         for li in range(len(pc.slot.levels))]
        for pc in prepared
    ]

    def _worst(entries: tuple[_Entry, ...]) -> tuple[
        float, tuple[float, float], tuple[tuple[int, int], ...]
    ]:
        """穷举偏差情景，返回 (最坏模长, 最坏残余向量, 各配重偏差符号)。

        情景按“配重顺序 → 质量符号(-1,0,+1) → 方位符号(-1,0,+1)”字典序
        枚举。首个情景无条件采纳；之后仅当模长**严格且显著**（超过
        TIE_EPS）更大才替换，因此情景并列（包括零偏差下所有情景模长都
        约等于 0）时保留符号字典序最小者。方案最多两个配重，直接展开
        k=0/1/2 三层循环，避免热路径上的闭包调用开销。
        """

        if not entries:
            return original_mag, (original_x, original_y), ()

        codes_table = _SCENARIO_CODES
        hypot = math.hypot

        if len(entries) == 1:
            e0 = entries[0]
            per0 = scen_flat[e0.slot_index][e0.level_index]
            ox, oy = original_x, original_y
            vx0, vy0 = per0[0]
            rx, ry = ox + vx0, oy + vy0
            worst_mag = hypot(rx, ry)
            worst_x, worst_y = rx, ry
            worst_code = codes_table[0]
            for flat0 in range(1, 9):
                vx0, vy0 = per0[flat0]
                rx, ry = ox + vx0, oy + vy0
                mag = hypot(rx, ry)
                if mag - worst_mag > TIE_EPS:
                    worst_mag, worst_x, worst_y = mag, rx, ry
                    worst_code = codes_table[flat0]
            return worst_mag, (worst_x, worst_y), (worst_code,)

        e0, e1 = entries
        per0 = scen_flat[e0.slot_index][e0.level_index]
        per1 = scen_flat[e1.slot_index][e1.level_index]
        ox, oy = original_x, original_y
        vx0, vy0 = per0[0]
        vx1, vy1 = per1[0]
        rx, ry = ox + vx0 + vx1, oy + vy0 + vy1
        worst_mag = hypot(rx, ry)
        worst_x, worst_y = rx, ry
        worst_code0 = worst_code1 = codes_table[0]
        for flat0 in range(9):
            vx0, vy0 = per0[flat0]
            bx, by = ox + vx0, oy + vy0
            code0 = codes_table[flat0]
            for flat1 in range(9):
                vx1, vy1 = per1[flat1]
                rx, ry = bx + vx1, by + vy1
                mag = hypot(rx, ry)
                if mag - worst_mag > TIE_EPS:
                    worst_mag, worst_x, worst_y = mag, rx, ry
                    worst_code0, worst_code1 = code0, codes_table[flat1]
        return worst_mag, (worst_x, worst_y), (worst_code0, worst_code1)

    def evaluate(entries: tuple[_Entry, ...]) -> tuple[
        _Key, tuple[float, float], tuple[tuple[int, int], ...]
    ]:
        """计算一个方案的择优键、最坏残余向量与最坏偏差符号组合。"""

        nx = original_x
        ny = original_y
        total_mass = 0.0
        vec = [0.0] * n_slots
        for e in entries:
            m, vx, vy = prepared[e.slot_index].nominal[e.level_index]
            nx += vx
            ny += vy
            total_mass += m
            vec[e.slot_index] = m
        nominal_mag = math.hypot(nx, ny)

        worst_mag, worst_xy, worst_codes = _worst(entries)

        key: _Key = (worst_mag, nominal_mag, total_mass, tuple(vec))
        return key, worst_xy, worst_codes

    # ---- 零配重方案（任何输入下恒为可行候选；偏差情景只有 1 个） ----
    best_key: _Key = (original_mag, original_mag, 0.0, (0.0,) * n_slots)
    best_entries: tuple[_Entry, ...] = ()
    best_worst = (original_x, original_y)
    best_combo: tuple[tuple[int, int], ...] = ()
    evaluated = 1
    max_scenarios = 1

    def consider(entries: tuple[_Entry, ...]) -> None:
        nonlocal best_key, best_entries, best_worst, best_combo
        nonlocal evaluated, max_scenarios
        key, worst_xy, worst_codes = evaluate(entries)
        evaluated += 1
        scenarios = 9 ** len(entries)
        if scenarios > max_scenarios:
            max_scenarios = scenarios
        if _strictly_better(key, best_key):
            best_key = key
            best_entries = entries
            best_worst = worst_xy
            best_combo = worst_codes

    # ---- 单槽方案：槽位顺序（= ID 升序）、质量升序 ----
    for i, pc in enumerate(prepared):
        for li in range(len(pc.slot.levels)):
            consider((_Entry(i, li),))

    # ---- 两槽方案：i < j；两槽质量档位按 (前槽档, 后槽档) 字典序升序 ----
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            for li in range(len(prepared[i].slot.levels)):
                for lj in range(len(prepared[j].slot.levels)):
                    consider((_Entry(i, li), _Entry(j, lj)))

    # ---- 组装结果（best_entries 已按槽位 ID 升序，直接映射输出） ----
    weights_out: list[ChosenWeight] = []
    combo_out: list[WeightScenario] = []
    for pos, e in enumerate(best_entries):
        sm_code, sa_code = best_combo[pos]
        pc = prepared[e.slot_index]
        s = pc.slot
        m = s.levels[e.level_index]
        worst_m = pc.actual_mass[e.level_index][sm_code + 1][sa_code + 1]
        worst_az = pc.actual_azimuth[e.level_index][sm_code + 1][sa_code + 1]
        weights_out.append(ChosenWeight(
            slot_id=s.id,
            radius=s.radius,
            nominal_mass=m,
            nominal_azimuth_deg=s.azimuth_deg,
            worst_mass=worst_m,
            worst_azimuth_deg=worst_az,
        ))
        combo_out.append(WeightScenario(
            slot_id=s.id,
            mass=worst_m,
            azimuth_deg=worst_az,
            mass_sign=sm_code,
            azimuth_sign=sa_code,
        ))

    nominal_x = original_x + sum(
        _vec(w.nominal_mass, w.radius, w.nominal_azimuth_deg)[0]
        for w in weights_out
    )
    nominal_y = original_y + sum(
        _vec(w.nominal_mass, w.radius, w.nominal_azimuth_deg)[1]
        for w in weights_out
    )
    worst_mag, nominal_mag, total_mass, mass_vec = best_key

    return BalancePlan(
        original_x=original_x,
        original_y=original_y,
        original_magnitude=original_mag,
        weights=tuple(weights_out),
        total_balance_mass=total_mass,
        nominal_x=nominal_x,
        nominal_y=nominal_y,
        nominal_magnitude=nominal_mag,
        worst_x=best_worst[0],
        worst_y=best_worst[1],
        worst_magnitude=worst_mag,
        worst_combo=tuple(combo_out),
        residual_limit=residual_moment_limit,
        # 残余模长不超过限值即达标（差值 <= 1e-9 视为并列；限值为 0 时
        # 浮点残差量级的残余同样判达标）
        compliant=not (
            worst_mag > residual_moment_limit
            and worst_mag - residual_moment_limit > TIE_EPS
        ),
        evaluated_plans=evaluated,
        max_scenarios_per_plan=max_scenarios,
        mass_vector=mass_vec,
        slot_ids=tuple(s.id for s in slots),
    )
