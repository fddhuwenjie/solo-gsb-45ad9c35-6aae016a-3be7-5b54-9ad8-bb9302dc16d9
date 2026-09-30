"""圆顶检修配重换装顺序规划：安全约束下的最短动作序列搜索。

场景
----

圆顶检修时需要把当前在位的一组**可操作配重**换装为另一组目标配重：

- ``fixed_parts``：固定部件（不可拆装），质量矩与质量始终在底座上；
- 每件可操作配重具有唯一 ID、质量、半径与方位，方位沿用
  ``planner.balance`` 的约定：**北为 0°、向东递增**，
  质量矩 ``m·r·(sin a, cos a)``（x=东向, y=北向）；
- 一次只能**装上（install）一件**当前不在位的配重，或
  **拆下（remove）一件**当前在位的配重；
- 目标集合之外的配重允许**临时**使用，但换装结束时必须全部拆除
  （终态恰好等于目标 ID 集合）。

每步结束（含起止状态）后都必须满足：

1. 在位总质量（固定部件 + 在位配重）不超过 ``installed_mass_limit``；
2. 残余质量矩模长 ``|Σ m·r·(sin a, cos a)|`` 不超过
   ``residual_moment_limit``。

搜索
----

状态空间为全部配重在位与否的位掩码（至多 12 件 → 4096 个状态）。
仅满足安全约束的状态可通行；一次装/拆即图上一条边，因此动作数最少
的安全路径 = 状态图上的无权**最短路**（逐层 BFS）。

同长度（同为最短路径）时依次按以下键择优（差值不超过
``TIE_EPS = 1e-9`` 视为并列）：

1. 路径上各步结束残余质量矩模长的**最大值**（中间残余较小者优；
   起止状态对所有候选相同，实际由中间状态决定）；
2. 动作序列字典序：动作类型 ``install < remove``（英文字典序），
   同类型按配重 ID 字符串升序。

逐层 BFS 中，每层从上一层已定型的全部路径做一次完整扫描后才定型本层
标签，因此同层内相互发现的等长路径也会参与择优，保证上述并列键全局
最优。

无路情形
--------

起点或终点本身越限直接抛出 :class:`WeightSequenceValidationError`
（HTTP 422）。其余无路情形（目标可达性失败）返回 HTTP 200、
``feasible=false``，并报告：

- ``reachable_states``：从起点可达的合法状态数（含起点）；
- ``nearest_blocked_action``：从可达状态出发、指向越限状态的被阻断
  动作中“最近一步”——依次按**质量矩超限量**、**总质量超限量**、
  动作字典序（install < remove，再按 ID 升序）取最小者；超限量为
  非负量（越限量，未越限记 0）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

# 与 planner.balance 一致的并列/边界容差：模长/质量不超过限值 1e-9
# 以内视为达标；择优键差值不超过它视为并列。
TIE_EPS = 1e-9

# 可操作配重件数上限（由请求模型 app.models 强制）。
MAX_WEIGHTS = 12

# 动作类型字典序：install < remove（英文按字母序）。
ACTION_INSTALL = "install"
ACTION_REMOVE = "remove"


class WeightSequenceValidationError(Exception):
    """起点或终点状态本身越限（或其它规划前语义非法）→ HTTP 422。"""

    def __init__(self, payload: dict):
        super().__init__(payload.get("detail", "配重换装请求非法"))
        self.payload = payload


@dataclass(frozen=True)
class WeightSpec:
    """一件可操作配重：ID、质量、半径、方位（度）与预计算质量矩。"""

    id: str
    mass: float
    radius: float
    azimuth_deg: float
    moment_x: float
    moment_y: float


@dataclass(frozen=True)
class WeightAction:
    """单步动作（拆装一件）及其后状态。"""

    index: int  # 1 基步号
    action: str  # install / remove
    weight_id: str
    installed_ids: tuple[str, ...]  # 该步结束后的在位 ID（升序）
    total_mass: float
    residual_east: float
    residual_north: float
    residual_magnitude: float
    temporary: bool  # 本步配重是否为临时件（起点不在位且目标外）


@dataclass(frozen=True)
class StateSnapshot:
    """某个掩码状态的完整物理量。"""

    mask: int
    installed_ids: tuple[str, ...]
    total_mass: float
    residual_east: float
    residual_north: float
    residual_magnitude: float
    legal: bool


@dataclass(frozen=True)
class BlockedAction:
    """无路诊断：从可达状态出发的最近一步被阻断动作。"""

    action: str
    weight_id: str
    from_depth: int  # 来源状态距起点的步数
    from_installed_ids: tuple[str, ...]
    resulting_installed_ids: tuple[str, ...]
    resulting_total_mass: float
    resulting_residual_east: float
    resulting_residual_north: float
    resulting_residual_magnitude: float
    moment_excess: float
    mass_excess: float


@dataclass(frozen=True)
class WeightSequencePlan:
    """换装规划结果（有解或无路诊断）。"""

    feasible: bool
    fixed_mass: float
    fixed_moment_x: float
    fixed_moment_y: float
    fixed_moment_magnitude: float
    installed_mass_limit: float
    residual_moment_limit: float
    initial: StateSnapshot
    target: StateSnapshot
    target_ids: tuple[str, ...]
    steps: tuple[WeightAction, ...] = ()
    temporary_weight_ids: tuple[str, ...] = ()
    reachable_states: int = 0
    max_intermediate_magnitude: float = 0.0
    nearest_blocked_action: BlockedAction | None = None
    explored_states: int = field(default=0)


def _moment(mass: float, radius: float, azimuth_deg: float) -> tuple[float, float]:
    """质量矩二维分量（东 x, 北 y）：方位北 0° 向东递增。"""

    a = math.radians(azimuth_deg)
    return mass * radius * math.sin(a), mass * radius * math.cos(a)


def _excess(value: float, limit: float) -> float:
    """越限量：超出限值的非负量；1e-9 边界容差内视为未越限。"""

    over = value - limit
    return over if over > TIE_EPS else 0.0


def _snapshot(
    mask: int,
    weights: list[WeightSpec],
    fixed_mass: float,
    fixed_x: float,
    fixed_y: float,
) -> StateSnapshot:
    """汇总某掩码状态的在位 ID、总质量、残余向量与模长（不判合法性）。"""

    ids = tuple(w.id for k, w in enumerate(weights) if mask & (1 << k))
    mass = fixed_mass
    rx, ry = fixed_x, fixed_y
    for k, w in enumerate(weights):
        if mask & (1 << k):
            mass += w.mass
            rx += w.moment_x
            ry += w.moment_y
    return StateSnapshot(
        mask=mask,
        installed_ids=ids,
        total_mass=mass,
        residual_east=rx,
        residual_north=ry,
        residual_magnitude=math.hypot(rx, ry),
        legal=False,  # 由调用方依据限值判定
    )


def plan_weight_sequence(
    fixed_parts: list[tuple[float, float, float]],
    weight_items: list[tuple[str, float, float, float]],
    current_ids: set[str],
    target_ids: set[str],
    installed_mass_limit: float,
    residual_moment_limit: float,
) -> WeightSequencePlan:
    """在安全约束下搜索动作最少的换装路径。

    参数
    ----
    fixed_parts:
        固定部件 ``(质量, 半径, 方位角)`` 三元组列表。
    weight_items:
        可操作配重 ``(ID, 质量, 半径, 方位角)`` 列表；调用方需保证
        ID 非空、唯一且件数 ≤ 12，方位单位为度。
    current_ids / target_ids:
        起点 / 终点在位配重 ID 集合；调用方需保证均为已知 ID。

    起点或终点越限时抛出 :class:`WeightSequenceValidationError`。
    """

    # 内部一律按 ID 升序排列配重：动作枚举顺序即 ID 升序，使字典序
    # 比较与结果可复现，且与请求数组顺序无关。
    weights: list[WeightSpec] = []
    for wid, mass, radius, azimuth in sorted(weight_items, key=lambda w: w[0]):
        mx, my = _moment(mass, radius, azimuth)
        weights.append(WeightSpec(
            id=wid, mass=mass, radius=radius, azimuth_deg=azimuth,
            moment_x=mx, moment_y=my,
        ))
    n = len(weights)
    index_of = {w.id: k for k, w in enumerate(weights)}

    fixed_mass = sum(m for m, _r, _a in fixed_parts)
    fixed_x = fixed_y = 0.0
    for mass, radius, azimuth in fixed_parts:
        vx, vy = _moment(mass, radius, azimuth)
        fixed_x += vx
        fixed_y += vy
    fixed_mag = math.hypot(fixed_x, fixed_y)

    start_mask = sum(1 << index_of[wid] for wid in current_ids)
    target_mask = sum(1 << index_of[wid] for wid in target_ids)
    target_set = target_ids
    current_set = set(current_ids)

    def snapshot(mask: int) -> StateSnapshot:
        snap = _snapshot(mask, weights, fixed_mass, fixed_x, fixed_y)
        legal_now = (
            _excess(snap.total_mass, installed_mass_limit) == 0.0
            and _excess(snap.residual_magnitude, residual_moment_limit) == 0.0
        )
        return replace(snap, legal=legal_now)

    initial = snapshot(start_mask)
    target = snapshot(target_mask)

    # 起点 / 终点本身越限 → HTTP 422（起点优先报告）。
    if not initial.legal:
        raise WeightSequenceValidationError(_illegal_endpoint_payload(
            reason="current_state_violates_limits",
            endpoint="current",
            snapshot=initial,
            mass_limit=installed_mass_limit,
            moment_limit=residual_moment_limit,
        ))
    if not target.legal:
        raise WeightSequenceValidationError(_illegal_endpoint_payload(
            reason="target_state_violates_limits",
            endpoint="target",
            snapshot=target,
            mass_limit=installed_mass_limit,
            moment_limit=residual_moment_limit,
        ))

    # 预计算全部掩码状态（至多 4096 个）的物理量与合法性。
    n_masks = 1 << n
    mass_of = [0.0] * n_masks
    x_of = [0.0] * n_masks
    y_of = [0.0] * n_masks
    legal = [False] * n_masks
    for mask in range(n_masks):
        mass = fixed_mass
        rx, ry = fixed_x, fixed_y
        for k, w in enumerate(weights):
            if mask & (1 << k):
                mass += w.mass
                rx += w.moment_x
                ry += w.moment_y
        mass_of[mask] = mass
        x_of[mask] = rx
        y_of[mask] = ry
        mag = math.hypot(rx, ry)
        legal[mask] = (
            _excess(mass, installed_mass_limit) == 0.0
            and _excess(mag, residual_moment_limit) == 0.0
        )

    # ---- 逐层 BFS：标签 = (路径上最大残余模长, 动作序列) ----
    # distance[mask]：距起点层数（动作数）；-1 表示尚未可达。
    distance = [-1] * n_masks
    # labels[mask] = (max_residual, ((action, weight_index), ...))
    labels: dict[int, tuple[float, tuple[tuple[str, int], ...]]] = {}
    distance[start_mask] = 0
    labels[start_mask] = (initial.residual_magnitude, ())

    frontier = [start_mask]
    d = 0
    while frontier:
        next_candidates: dict[int, tuple[float, tuple[tuple[str, int], ...]]] = {}
        for mask in frontier:
            cur_max, cur_actions = labels[mask]
            # 动作枚举：先全部 install（位 0），再全部 remove（位 1）；
            # 各类内部按位序（= ID 升序），与标签字典序完全一致。
            for k, w in enumerate(weights):
                bit = 1 << k
                if not (mask & bit):
                    action = ACTION_INSTALL
                    nmask = mask | bit
                else:
                    action = ACTION_REMOVE
                    nmask = mask ^ bit
                if not legal[nmask]:
                    continue
                if distance[nmask] != -1:
                    # 只能到达更浅/同层已定型的状态：更浅无需更新，
                    # 同层已在本轮扫描中定型的也不回改。
                    continue
                new_actions = cur_actions + ((action, k),)
                new_max = max(cur_max, math.hypot(x_of[nmask], y_of[nmask]))
                cand = (new_max, new_actions)
                old = next_candidates.get(nmask)
                if old is None or _label_less(cand, old):
                    next_candidates[nmask] = cand
        # 本层全部扫描完毕后定型，保证同层互达的等长路径参与择优。
        next_frontier: list[int] = []
        for nmask, lab in next_candidates.items():
            distance[nmask] = d + 1
            labels[nmask] = lab
            next_frontier.append(nmask)
        frontier = next_frontier
        d += 1

    reachable = [m for m in range(n_masks) if distance[m] != -1]

    if distance[target_mask] != -1:
        max_mag, action_idx = labels[target_mask]
        steps = _rebuild_steps(
            action_idx, weights, start_mask,
            mass_of, x_of, y_of, target_set, current_set,
        )
        # “临时配重”= 起点不在位、目标也不含（中途借来且最终拆除）的配重；
        # 起点在位而终点拆除的旧配重属于正常换装对象，不计为临时。
        used_ids = {weights[k].id for (_a, k) in action_idx}
        temporary_ids = tuple(sorted(
            w_id for w_id in used_ids
            if w_id not in target_set and w_id not in current_set
        ))
        return WeightSequencePlan(
            feasible=True,
            fixed_mass=fixed_mass,
            fixed_moment_x=fixed_x,
            fixed_moment_y=fixed_y,
            fixed_moment_magnitude=fixed_mag,
            installed_mass_limit=installed_mass_limit,
            residual_moment_limit=residual_moment_limit,
            initial=initial,
            target=target,
            target_ids=tuple(sorted(target_set)),
            steps=steps,
            temporary_weight_ids=temporary_ids,
            reachable_states=len(reachable),
            max_intermediate_magnitude=max_mag,
            nearest_blocked_action=None,
            explored_states=n_masks,
        )

    # ---- 无路：枚举从可达状态出发、指向越限状态的被阻断动作 ----
    blocked: list[BlockedAction] = []
    for mask in reachable:
        from_snap_ids = tuple(
            weights[k].id for k in range(n) if mask & (1 << k)
        )
        for k, w in enumerate(weights):
            bit = 1 << k
            if mask & bit:
                action = ACTION_REMOVE
                nmask = mask ^ bit
            else:
                action = ACTION_INSTALL
                nmask = mask | bit
            if legal[nmask]:
                continue
            rmag = math.hypot(x_of[nmask], y_of[nmask])
            blocked.append(BlockedAction(
                action=action,
                weight_id=w.id,
                from_depth=distance[mask],
                from_installed_ids=from_snap_ids,
                resulting_installed_ids=tuple(
                    weights[j].id for j in range(n) if nmask & (1 << j)
                ),
                resulting_total_mass=mass_of[nmask],
                resulting_residual_east=x_of[nmask],
                resulting_residual_north=y_of[nmask],
                resulting_residual_magnitude=rmag,
                moment_excess=_excess(rmag, residual_moment_limit),
                mass_excess=_excess(mass_of[nmask], installed_mass_limit),
            ))

    # “最近一步”：质量矩超限量 → 总质量超限量 → 动作字典序
    # （install < remove，再按 ID 升序）；其后以来源步数、来源在位 ID
    # 收束完全并列，保证结果唯一可复现。
    nearest = min(
        blocked,
        key=lambda b: (
            b.moment_excess,
            b.mass_excess,
            (b.action, b.weight_id),
            b.from_depth,
            b.from_installed_ids,
        ),
    )

    return WeightSequencePlan(
        feasible=False,
        fixed_mass=fixed_mass,
        fixed_moment_x=fixed_x,
        fixed_moment_y=fixed_y,
        fixed_moment_magnitude=fixed_mag,
        installed_mass_limit=installed_mass_limit,
        residual_moment_limit=residual_moment_limit,
        initial=initial,
        target=target,
        target_ids=tuple(sorted(target_set)),
        steps=(),
        temporary_weight_ids=(),
        reachable_states=len(reachable),
        max_intermediate_magnitude=0.0,
        nearest_blocked_action=nearest,
        explored_states=n_masks,
    )


def _label_less(
    a: tuple[float, tuple[tuple[str, int], ...]],
    b: tuple[float, tuple[tuple[str, int], ...]],
) -> bool:
    """路径标签并列感知比较：最大残余模长小者优，再比动作序列字典序。

    最大残余差值不超过 TIE_EPS 视为并列；动作类型与索引均为离散量，
    序列按 (action 字符串, 位序) 逐对精确比较。
    """

    amax, aseq = a
    bmax, bseq = b
    if abs(amax - bmax) > TIE_EPS:
        return amax < bmax
    return aseq < bseq


def _rebuild_steps(
    action_idx: tuple[tuple[str, int], ...],
    weights: list[WeightSpec],
    start_mask: int,
    mass_of: list[float],
    x_of: list[float],
    y_of: list[float],
    target_set: set[str],
    current_set: set[str],
) -> tuple[WeightAction, ...]:
    """沿标签动作序列从起点重放，生成逐步响应记录。"""

    # 临时配重：起点不在位且目标外（中途借来、最终拆除）。
    temporary_ids = {
        w.id for w in weights
        if w.id not in target_set and w.id not in current_set
    }
    mask = start_mask
    steps: list[WeightAction] = []
    for i, (action, k) in enumerate(action_idx, start=1):
        w = weights[k]
        if action == ACTION_INSTALL:
            mask |= 1 << k
        else:
            mask ^= 1 << k
        rx, ry = x_of[mask], y_of[mask]
        steps.append(WeightAction(
            index=i,
            action=action,
            weight_id=w.id,
            installed_ids=tuple(
                weights[j].id for j in range(len(weights)) if mask & (1 << j)
            ),
            total_mass=mass_of[mask],
            residual_east=rx,
            residual_north=ry,
            residual_magnitude=math.hypot(rx, ry),
            temporary=w.id in temporary_ids,
        ))
    return tuple(steps)


def _illegal_endpoint_payload(
    *,
    reason: str,
    endpoint: str,
    snapshot: StateSnapshot,
    mass_limit: float,
    moment_limit: float,
) -> dict:
    """构造起点/终点越限的 422 响应体。"""

    return {
        "detail": (
            f"{'起点' if endpoint == 'current' else '终点'}状态越限："
            "在位总质量或残余质量矩模长超过限值"
        ),
        "reason": reason,
        "endpoint": endpoint,
        "installed_ids": list(snapshot.installed_ids),
        "total_mass": snapshot.total_mass,
        "installed_mass_limit": mass_limit,
        "mass_excess": _excess(snapshot.total_mass, mass_limit),
        "residual": {
            "east": snapshot.residual_east,
            "north": snapshot.residual_north,
            "magnitude": snapshot.residual_magnitude,
        },
        "residual_moment_limit": moment_limit,
        "moment_excess": _excess(snapshot.residual_magnitude, moment_limit),
    }
