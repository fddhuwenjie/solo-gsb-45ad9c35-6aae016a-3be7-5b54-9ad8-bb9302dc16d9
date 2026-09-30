"""圆顶检修配重换装顺序规划：安全子集状态图上的 BFS 最短动作路径。

状态空间
--------

每件可操作配重 ``w_i``（唯一 ID、质量、半径、方位）只有“在位 / 不在位”
两种状态，全体配重的在位子集用位掩码表示（``n <= 12``，最多 4096 个状态）。
固定部件始终在位，其质量与质量矩并入每个状态的在位总量：

- 在位总质量 ``M = Σ固定部件质量 + Σ在位配重质量``；
- 残余质量矩 ``R = Σ固定部件矩 + Σ在位配重矩``，矩的方位约定与
  ``planner.balance`` 相同：北为 0°、向东递增，分量
  ``(x=东向, y=北向) = m·r·(sin a, cos a)``。

掩码状态安全当且仅当（边界 ``TIE_EPS`` 内视为达标）：

- ``M <= installed_mass_limit``；
- ``|R| <= residual_moment_limit``。

动作与路径
----------

每一步对恰好一件配重执行 **install**（不在位 → 在位）或 **remove**
（在位 → 不在位），动作结束后的状态必须安全。路径自当前在位集合
``current`` 出发，到目标集合 ``target`` 结束；目标集合之外的配重允许
中途临时在位，但终点掩码等于 ``target`` 保证它们最终全部拆除。

搜索与并列规则
--------------

在安全状态构成的无向超立方体子图上做分层 BFS，动作步数最少者优先。
同一掩码可能由同层多个父状态到达，保留路径键最优者：

1. 路径上各步结束残余模长的最大值（小者优，``TIE_EPS`` 内并列）；
2. 动作序列字典序：逐项比较 ``(动作类型, 配重 ID)``，动作类型
   ``install < remove``，同类按配重 ID 升序。

起点或终点本身越限属请求语义非法，抛 :class:`SwapValidationError`
（HTTP 422）。终点不可达时给出从任意可达状态出发、被安全约束阻断的
全部动作中“最近的一步”：依次按质量矩超限量、总质量超限量、
动作字典序（再以源状态在位 ID 字典序兜底）取最小者。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 并列 / 达标判定容差，与 planner.balance 保持一致
TIE_EPS = 1e-9

# 可操作配重件数上限（请求模型同步强制）
MAX_WEIGHTS = 12

# 动作类型代码：字典序 install < remove
INSTALL = 0
REMOVE = 1
ACTION_TYPE_NAMES = {INSTALL: "install", REMOVE: "remove"}


class SwapValidationError(Exception):
    """请求语义非法（起点/终点越限、ID 引用未知等），映射为 HTTP 422。"""

    def __init__(self, payload: dict):
        super().__init__(str(payload.get("detail", "")))
        self.payload = payload


@dataclass(frozen=True)
class Weight:
    """一件可操作配重及其质量矩向量。"""

    id: str
    mass: float
    radius: float
    azimuth_deg: float
    mx: float
    my: float


@dataclass(frozen=True)
class SwapStep:
    """换装路径上的一步（动作结束时刻的完整状态）。"""

    index: int
    action_type: int
    weight_id: str
    installed_ids: tuple[str, ...]
    total_mass: float
    residual_x: float
    residual_y: float
    residual_magnitude: float


@dataclass(frozen=True)
class BlockedStep:
    """从可达状态出发但落点越限、被安全约束阻断的动作。"""

    action_type: int
    weight_id: str
    from_installed_ids: tuple[str, ...]
    total_mass: float
    mass_excess: float
    residual_x: float
    residual_y: float
    residual_magnitude: float
    moment_excess: float


@dataclass(frozen=True)
class SwapPlan:
    """换装搜索结果。"""

    feasible: bool
    steps: tuple[SwapStep, ...]
    reachable_states: int
    nearest_blocked: BlockedStep | None
    # 起点 / 终点测量值（回显与 422 诊断用）
    start_mass: float
    start_x: float
    start_y: float
    start_magnitude: float
    goal_mass: float
    goal_x: float
    goal_y: float
    goal_magnitude: float
    weight_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Node:
    """BFS 已达状态的最优路径信息。"""

    # 自起点到本状态的完整动作序列 ((类型代码, 配重 ID), ...)
    actions: tuple[tuple[int, str], ...]
    # 路径上各步结束残余模长的最大值
    max_residual: float
    parent: int
    parent_action_type: int
    parent_action_index: int


def _vec(mass: float, radius: float, azimuth_deg: float) -> tuple[float, float]:
    a = math.radians(azimuth_deg)
    return mass * radius * math.sin(a), mass * radius * math.cos(a)


def _eps_smaller(a: float, b: float) -> bool:
    """差值不超过 TIE_EPS 视为并列；两个近零值同样并列。"""

    if abs(a) <= TIE_EPS and abs(b) <= TIE_EPS:
        return False
    return a < b and b - a > TIE_EPS


def _better_path(
    cand_max: float,
    cand_actions: tuple[tuple[int, str], ...],
    incumbent: _Node,
) -> bool:
    """同层候选路径是否优于已记录路径：最大中间残余较小，再比动作字典序。"""

    if _eps_smaller(cand_max, incumbent.max_residual):
        return True
    if _eps_smaller(incumbent.max_residual, cand_max):
        return False
    return cand_actions < incumbent.actions


def _ids_of(mask: int, ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(ids[i] for i in range(len(ids)) if mask & (1 << i))


def plan_weight_swap(
    fixed_parts: list[tuple[float, float, float]],
    weights: list[Weight],
    current_ids: list[str],
    target_ids: list[str],
    installed_mass_limit: float,
    residual_moment_limit: float,
) -> SwapPlan:
    """搜索 current → target 的安全最短换装路径。

    参数
    ----
    fixed_parts:
        固定部件 ``(质量, 半径, 方位角)`` 三元组，始终在位。
    weights:
        可操作配重（调用方按 ID 去重；本函数内部按 ID 升序编号）。
    current_ids / target_ids:
        起点 / 终点在位配重 ID 列表（须为 weights 的子集，内部去重由
        请求模型保证）。

    起点或终点越限抛 :class:`SwapValidationError`。
    """

    # 内部一律按 ID 升序编号，结果与请求数组顺序无关。
    weights = sorted(weights, key=lambda w: w.id)
    ids = tuple(w.id for w in weights)
    id_to_index = {wid: i for i, wid in enumerate(ids)}
    n = len(weights)
    size = 1 << n

    def _mask(name_ids: list[str]) -> int:
        mask = 0
        for wid in name_ids:
            if wid not in id_to_index:
                raise SwapValidationError(
                    {"detail": f"未知配重 ID：{wid!r}", "unknown_id": wid}
                )
            mask |= 1 << id_to_index[wid]
        return mask

    start_mask = _mask(current_ids)
    goal_mask = _mask(target_ids)

    # 固定部件的质量与质量矩
    fixed_mass = 0.0
    fixed_x = fixed_y = 0.0
    for mass, radius, azimuth in fixed_parts:
        fixed_mass += mass
        vx, vy = _vec(mass, radius, azimuth)
        fixed_x += vx
        fixed_y += vy

    # 逐掩码预计算在位配重质量 / 矩分量（lowbit 递推），安全检查 O(1)。
    w_mass = [w.mass for w in weights]
    w_x = [w.mx for w in weights]
    w_y = [w.my for w in weights]
    mass_arr = [0.0] * size
    x_arr = [0.0] * size
    y_arr = [0.0] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        prev = mask ^ low
        mass_arr[mask] = mass_arr[prev] + w_mass[i]
        x_arr[mask] = x_arr[prev] + w_x[i]
        y_arr[mask] = y_arr[prev] + w_y[i]

    def measure(mask: int) -> tuple[float, float, float, float]:
        m = fixed_mass + mass_arr[mask]
        rx = fixed_x + x_arr[mask]
        ry = fixed_y + y_arr[mask]
        return m, rx, ry, math.hypot(rx, ry)

    def violations(mass: float, mag: float) -> tuple[float, float]:
        """返回 (质量矩超限量, 总质量超限量)，未越限项为 0。"""

        moment_excess = max(0.0, mag - residual_moment_limit)
        mass_excess = max(0.0, mass - installed_mass_limit)
        return moment_excess, mass_excess

    def is_safe(mass: float, mag: float) -> bool:
        me, xe = violations(mass, mag)
        return me <= TIE_EPS and xe <= TIE_EPS

    start_mass, sx, sy, start_mag = measure(start_mask)
    goal_mass, gx, gy, goal_mag = measure(goal_mask)

    for endpoint, mask, mass, mag in (
        ("start", start_mask, start_mass, start_mag),
        ("goal", goal_mask, goal_mass, goal_mag),
    ):
        moment_excess, mass_excess = violations(mass, mag)
        if moment_excess > TIE_EPS or mass_excess > TIE_EPS:
            rx, ry = (sx, sy) if endpoint == "start" else (gx, gy)
            raise SwapValidationError(
                {
                    "detail": (
                        f"{'起点' if endpoint == 'start' else '终点'}状态越限："
                        "在位总质量或残余质量矩超出限值"
                    ),
                    "endpoint": endpoint,
                    "installed_ids": list(_ids_of(mask, ids)),
                    "total_mass": mass,
                    "installed_mass_limit": installed_mass_limit,
                    "mass_excess": mass_excess,
                    "residual": {"east": rx, "north": ry},
                    "residual_magnitude": mag,
                    "residual_moment_limit": residual_moment_limit,
                    "moment_excess": moment_excess,
                }
            )

    # ---- 分层 BFS ----
    visited: dict[int, _Node] = {
        start_mask: _Node(
            actions=(),
            max_residual=0.0,
            parent=-1,
            parent_action_type=INSTALL,
            parent_action_index=-1,
        )
    }
    frontier = [start_mask]
    blocked: list[BlockedStep] = []

    while frontier:
        # 本层候选先落到 next_layer，层末再并入 visited，保证同层父状态
        # 全部参与并列比较后，下一层才开始扩展。
        next_layer: dict[int, _Node] = {}
        for mask in frontier:
            node = visited[mask]

            # 动作枚举顺序即字典序：先全部 install（ID 升序），再全部
            # remove（ID 升序）；显式比较路径键，枚举顺序仅影响并列兜底。
            candidates: list[tuple[int, int, int]] = []
            for i in range(n):
                bit = 1 << i
                if not mask & bit:
                    candidates.append((INSTALL, i, mask | bit))
            for i in range(n):
                bit = 1 << i
                if mask & bit:
                    candidates.append((REMOVE, i, mask ^ bit))

            for action_type, i, child in candidates:
                if child in visited:
                    # 更短（或等长更早）层已达：不再视为候选，也不是阻断。
                    continue

                mass, rx, ry, mag = measure(child)
                moment_excess, mass_excess = violations(mass, mag)

                if moment_excess > TIE_EPS or mass_excess > TIE_EPS:
                    blocked.append(
                        BlockedStep(
                            action_type=action_type,
                            weight_id=ids[i],
                            from_installed_ids=_ids_of(mask, ids),
                            total_mass=mass,
                            mass_excess=mass_excess,
                            residual_x=rx,
                            residual_y=ry,
                            residual_magnitude=mag,
                            moment_excess=moment_excess,
                        )
                    )
                    continue

                cand_actions = node.actions + ((action_type, ids[i]),)
                cand_max = max(node.max_residual, mag)
                old = next_layer.get(child)
                if old is None or _better_path(cand_max, cand_actions, old):
                    next_layer[child] = _Node(
                        actions=cand_actions,
                        max_residual=cand_max,
                        parent=mask,
                        parent_action_type=action_type,
                        parent_action_index=i,
                    )

        for child, node in next_layer.items():
            visited[child] = node
        frontier = list(next_layer)

    reachable = len(visited)

    def _build_steps() -> tuple[SwapStep, ...]:
        # 自终点沿 parent 回溯再反转
        chain: list[tuple[int, int, int]] = []  # (动作类型, 配重下标, 动作后掩码)
        mask = goal_mask
        while mask != start_mask:
            node = visited[mask]
            chain.append((node.parent_action_type, node.parent_action_index, mask))
            mask = node.parent
        chain.reverse()

        steps: list[SwapStep] = []
        for k, (action_type, i, after_mask) in enumerate(chain, start=1):
            mass, rx, ry, mag = measure(after_mask)
            steps.append(
                SwapStep(
                    index=k,
                    action_type=action_type,
                    weight_id=ids[i],
                    installed_ids=_ids_of(after_mask, ids),
                    total_mass=mass,
                    residual_x=rx,
                    residual_y=ry,
                    residual_magnitude=mag,
                )
            )
        return tuple(steps)

    nearest: BlockedStep | None = None
    if goal_mask in visited:
        return SwapPlan(
            feasible=True,
            steps=_build_steps(),
            reachable_states=reachable,
            nearest_blocked=None,
            start_mass=start_mass,
            start_x=sx,
            start_y=sy,
            start_magnitude=start_mag,
            goal_mass=goal_mass,
            goal_x=gx,
            goal_y=gy,
            goal_magnitude=goal_mag,
            weight_ids=ids,
        )

    # 无路：在全部阻断动作中依次按质量矩超限量、总质量超限量、
    # 动作字典序（(类型, ID)）、源状态在位 ID 序列取最小者。
    def _blocked_less(a: BlockedStep, b: BlockedStep) -> bool:
        if _eps_smaller(a.moment_excess, b.moment_excess):
            return True
        if _eps_smaller(b.moment_excess, a.moment_excess):
            return False
        if _eps_smaller(a.mass_excess, b.mass_excess):
            return True
        if _eps_smaller(b.mass_excess, a.mass_excess):
            return False
        ak = (a.action_type, a.weight_id)
        bk = (b.action_type, b.weight_id)
        if ak != bk:
            return ak < bk
        return a.from_installed_ids < b.from_installed_ids

    for cand in blocked:
        if nearest is None or _blocked_less(cand, nearest):
            nearest = cand

    return SwapPlan(
        feasible=False,
        steps=(),
        reachable_states=reachable,
        nearest_blocked=nearest,
        start_mass=start_mass,
        start_x=sx,
        start_y=sy,
        start_magnitude=start_mag,
        goal_mass=goal_mass,
        goal_x=gx,
        goal_y=gy,
        goal_magnitude=goal_mag,
        weight_ids=ids,
    )
