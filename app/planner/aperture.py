"""视轴投影到偏心球形穹顶：前向射线-球面求交与开口包络。

坐标系
------
ENU 局部直角坐标（单位与穹顶半径相同）：``E`` 向东、``N`` 向北、``U`` 向上。
球形穹顶中心固定在穹顶地平面原点 ``(0, 0, 0)``，半径为 ``R``，因此：

- 视轴起点必须**严格位于球内**（``|o| < R``），否则前向射线可能无正根，
  接口返回 HTTP 422；
- 穹顶地平面为 ``z = 0``，交点 ``U`` 分量为负即落在地平面以下。

视线方向（方位角 ``az`` 北为 0°、向东递增；高度角 ``el`` 自地平面起算）：

    d = ( sin(az)·cos(el), cos(az)·cos(el), sin(el) )。

射线与球面 ``|o + t·d|² = R²`` 的最近正根（``t > 0``，即前向首个交点）
采用稳定形式求解，避免近切线情形下的相消误差。

开口包络
--------
对每个样本的标称视轴与误差矩形四角
``(az ± δaz, el ± δel)`` 共 5 条射线求交。5 个交点的球面方位构成圆上点集，
其**最小覆盖圆弧**通过“最大间隙法”求得：最大对侧间隙可以并列，并列时取
``center``（归一化到 [0, 360)）较小者；圆弧跨 0° 时以 ``center/width``
表示（``azimuth_crosses_zero = true``）。高度包络为 5 个交点球面高度的
``[min, max]``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FULL_TURN = 360.0

# 角度容差（度）：小于该值的角宽视为退化
ANGLE_EPS = 1e-9
# 球内判定的长度容差（|o| 距球面小于该值视为恰在面上 → 拒绝）
INSIDE_EPS = 1e-9


class ApertureValidationError(ValueError):
    """请求几何语义非法，应映射为 HTTP 422；payload 为可直接序列化的诊断字典。"""

    def __init__(self, payload: dict) -> None:
        super().__init__(payload.get("detail", "aperture validation error"))
        self.payload = payload


@dataclass(frozen=True)
class Vec3:
    e: float
    n: float
    u: float


@dataclass(frozen=True)
class Intersection:
    """单条前向射线与球面的交点。"""

    distance: float
    point: Vec3
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class Envelope:
    """5 个交点的开口包络。"""

    azimuth_center: float
    azimuth_width: float
    azimuth_crosses_zero: bool
    elevation_low: float
    elevation_high: float


def look_direction(azimuth_deg: float, elevation_deg: float) -> Vec3:
    """方位/高度（度）→ ENU 单位方向向量。"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    ce = math.cos(el)
    return Vec3(math.sin(az) * ce, math.cos(az) * ce, math.sin(el))


def ray_sphere_intersect(origin: Vec3, direction: Vec3, radius: float) -> float:
    """前向射线 ``origin + t·direction`` 与球 ``|x| = radius`` 的最近正根。

    调用方需保证 ``direction`` 为单位向量且 ``|origin| < radius``，
    此时判别式恒为正（起点严格在球内），唯一的前向正根是离开球面的出口。
    采用稳定求根（按 ``B = o·d`` 的符号选根），避免近切线时两根相消。
    """
    ox, oy, oz = origin.e, origin.n, origin.u
    dx, dy, dz = direction.e, direction.n, direction.u
    # |o + t d|^2 = R^2  →  t^2 + 2B t + c = 0（d 为单位向量），
    # B = o·d，c = |o|^2 - R^2 < 0（起点严格在球内），故两根异号，
    # t = -B ± sqrt(B^2 - c) 中唯一正根为前向出口。
    b = ox * dx + oy * dy + oz * dz
    norm = math.sqrt(ox * ox + oy * oy + oz * oz)
    # 直接写 |o|^2 - R^2 在起点贴近球面时会发生相消（两个 ~R^2 的数相减），
    # 改用因式 (|o| - R)(|o| + R) 计算 c，保留 |o| 与 R 之差的全部有效数字。
    c = (norm - radius) * (norm + radius)

    disc = b * b - c
    # 理论上 disc > 0（严格球内），仅在极端舍入下兜底，防止 sqrt 负数。
    s = math.sqrt(disc if disc > 0.0 else 0.0)
    # 数值稳定根，避免近切线（B > 0、s ≈ B）时 -B + s 的相消：
    # B >= 0 时正根 = -c / (B + s)（两根乘积为 c）；B < 0 时 s - B 本为
    # 两正数相加，无抵消，直接计算。
    if b >= 0.0:
        return -c / (b + s)
    return s - b


def spherical_azimuth(point: Vec3) -> float:
    """ENU 点的球面方位角（度，[0, 360)，北为 0°、向东递增）。"""
    a = math.degrees(math.atan2(point.e, point.n)) % FULL_TURN
    if a >= FULL_TURN - ANGLE_EPS:
        a = 0.0
    return a


def spherical_elevation(point: Vec3) -> float:
    """ENU 点的球面高度角（度，[-90, 90]）。"""
    horizontal = math.hypot(point.e, point.n)
    return math.degrees(math.atan2(point.u, horizontal))


def project_ray(origin: Vec3, azimuth_deg: float, elevation_deg: float,
                radius: float) -> Intersection:
    """求一条视轴射线与穹顶内表面的交点及其球面方位/高度。"""
    direction = look_direction(azimuth_deg, elevation_deg)
    t = ray_sphere_intersect(origin, direction, radius)
    point = Vec3(origin.e + t * direction.e,
                 origin.n + t * direction.n,
                 origin.u + t * direction.u)
    return Intersection(
        distance=t,
        point=point,
        azimuth=spherical_azimuth(point),
        elevation=spherical_elevation(point),
    )


def shortest_covering_arc(azimuths_deg: list[float]) -> tuple[float, float, bool]:
    """圆上点集的最小覆盖圆弧。

    返回 ``(center, width, crosses_zero)``：``center`` 归一化到 [0, 360)，
    ``width`` 为覆盖圆弧角宽（0..360，点集退化为单点时为 0）。
    采用最大间隙法：圆弧 = 圆周去掉最大的点间间隙；最大间隙并列时取
    ``center`` 较小者。间隙跨 0°（即被去掉的间隙跨过 0°，覆盖弧本身不跨
    0°）与不跨 0° 的候选统一按归一化后的 ``center`` 比较。
    """
    pts = sorted(a % FULL_TURN for a in azimuths_deg)
    m = len(pts)

    if m == 1 or pts[-1] - pts[0] <= ANGLE_EPS:
        return pts[0], 0.0, False

    # 各点之后到下一有序点的间隙；gaps[i] 是离开 pts[i] 的前向间隙。
    gaps = [pts[i + 1] - pts[i] for i in range(m - 1)]
    gaps.append(pts[0] + FULL_TURN - pts[-1])  # 末点绕回首点的间隙

    max_gap = max(gaps)

    def candidate(i: int) -> tuple[float, float, bool]:
        # 去掉从 pts[i] 跨到下一点（前向）的最大间隙后，覆盖弧从间隙的
        # 终点开始、到间隙的起点结束。
        start = pts[(i + 1) % m]
        end = pts[i]
        width = FULL_TURN - gaps[i]
        if i == m - 1:
            # 被去掉的是绕回间隙（跨 0°），覆盖弧为 [start, end]（不跨 0°）
            center = (start + end) / 2.0
            crosses = False
        else:
            # 覆盖弧从 start 前向绕到 end + 360，跨 0°
            center = ((start + (end + FULL_TURN)) / 2.0) % FULL_TURN
            crosses = True
        return center, width, crosses

    # 并列取 center 较小者
    best: tuple[float, float, bool] | None = None
    for i, g in enumerate(gaps):
        if g == max_gap:
            cand = candidate(i)
            if best is None or cand[0] < best[0]:
                best = cand
    assert best is not None
    return best


def aperture_envelope(intersections: list[Intersection]) -> Envelope:
    """由 5 个（标称 + 误差矩形四角）交点求开口包络。"""
    center, width, crosses = shortest_covering_arc(
        [hit.azimuth for hit in intersections]
    )
    elevations = [hit.elevation for hit in intersections]
    return Envelope(
        azimuth_center=center,
        azimuth_width=width,
        azimuth_crosses_zero=crosses,
        elevation_low=min(elevations),
        elevation_high=max(elevations),
    )


def ensure_origin_inside(origin: Vec3, radius: float) -> None:
    """视轴起点不在球内（含距球面不足 ``INSIDE_EPS``）→ 422 诊断。

    判据用 ``|o| + INSIDE_EPS < R``：恰在球面上（前向射线可能无正根或
    为零根）与数值上贴近球面的输入都拒绝；明显位于球内者通过。
    """
    norm = math.sqrt(max(
        origin.e ** 2 + origin.n ** 2 + origin.u ** 2, 0.0
    ))
    if not norm + INSIDE_EPS < radius:
        raise ApertureValidationError({
            "detail": "视轴起点不在穹顶球体内部（要求 |origin| < dome_radius）",
            "reason": "origin_outside_sphere",
            "origin": {"east": origin.e, "north": origin.n, "up": origin.u},
            "origin_radius": norm,
            "dome_radius": radius,
        })
