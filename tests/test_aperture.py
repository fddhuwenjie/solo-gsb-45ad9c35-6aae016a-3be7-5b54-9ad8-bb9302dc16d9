"""视轴投影到偏心球形穹顶：几何算法与 HTTP API 回归测试。"""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planner.aperture import (
    Vec3,
    aperture_envelope,
    ensure_origin_inside,
    project_ray,
    ray_sphere_intersect,
    shortest_covering_arc,
    ApertureValidationError,
)

client = TestClient(app)


def _sample(index, az, el, daz=0.0, dele=0.0):
    return {
        "index": index,
        "azimuth": az,
        "elevation": el,
        "azimuth_error": daz,
        "elevation_error": dele,
    }


def _payload(origin=(0.0, 0.0, 0.0), radius=10.0, samples=None):
    return {
        "dome_radius": radius,
        "origin": {"east": origin[0], "north": origin[1], "up": origin[2]},
        "samples": samples if samples is not None else [_sample(0, 0.0, 30.0)],
    }


# ---------------------------------------------------------------------------
# 几何算法
# ---------------------------------------------------------------------------


def test_centric_horizontal_ray_hits_at_radius():
    hit = project_ray(Vec3(0.0, 0.0, 0.0), 0.0, 0.0, 10.0)
    assert hit.distance == pytest.approx(10.0)
    assert hit.point.e == pytest.approx(0.0, abs=1e-9)
    assert hit.point.n == pytest.approx(10.0, abs=1e-9)
    assert hit.point.u == pytest.approx(0.0, abs=1e-9)
    assert hit.azimuth == pytest.approx(0.0, abs=1e-9)
    assert hit.elevation == pytest.approx(0.0, abs=1e-9)


def test_azimuth_north_zero_east_increasing():
    for az, (e, n) in [
        (0.0, (0.0, 1.0)),
        (90.0, (1.0, 0.0)),
        (180.0, (0.0, -1.0)),
        (270.0, (-1.0, 0.0)),
    ]:
        hit = project_ray(Vec3(0.0, 0.0, 0.0), az, 0.0, 5.0)
        assert hit.point.e == pytest.approx(5.0 * e, abs=1e-8)
        assert hit.point.n == pytest.approx(5.0 * n, abs=1e-8)
        assert hit.azimuth == pytest.approx(az % 360.0, abs=1e-8)


def test_eccentric_origin_exact_roots():
    # 起点 (1, 0, 2) 向北水平：b=0, t = sqrt(R^2 - |o|^2) = sqrt(95)
    origin = Vec3(1.0, 0.0, 2.0)
    hit = project_ray(origin, 0.0, 0.0, 10.0)
    t = math.sqrt(95.0)
    assert hit.distance == pytest.approx(t, abs=1e-9)
    assert hit.point.e == pytest.approx(1.0, abs=1e-9)
    assert hit.point.n == pytest.approx(t, abs=1e-9)
    assert hit.point.u == pytest.approx(2.0, abs=1e-9)
    # 交点确在球面上
    r2 = hit.point.e ** 2 + hit.point.n ** 2 + hit.point.u ** 2
    assert r2 == pytest.approx(100.0, abs=1e-8)
    # 球面方位/高度相对球心
    assert hit.azimuth == pytest.approx(math.degrees(math.atan2(1.0, t)), abs=1e-9)
    assert hit.elevation == pytest.approx(
        math.degrees(math.atan2(2.0, math.hypot(1.0, t))), abs=1e-9
    )


def test_nearest_positive_root_is_exit_point():
    # 任意随机方向与偏心起点：解析二次方程验证最近正根
    origin = Vec3(-2.3, 1.7, 0.6)
    az, el, R = 137.4, 28.9, 8.0
    hit = project_ray(origin, az, el, R)
    assert hit.distance > 0.0
    p = hit.point
    assert p.e ** 2 + p.n ** 2 + p.u ** 2 == pytest.approx(R ** 2, abs=1e-7)
    # 起点与交点连线与视线方向一致
    el_rad, az_rad = math.radians(el), math.radians(az)
    d = (
        math.sin(az_rad) * math.cos(el_rad),
        math.cos(az_rad) * math.cos(el_rad),
        math.sin(el_rad),
    )
    for got, want in zip((p.e - origin.e, p.n - origin.n, p.u - origin.u), d):
        assert got / hit.distance == pytest.approx(want, abs=1e-9)


def test_near_tangent_ray_stable_root():
    # 起点非常接近球面，射线朝接近“出口”方向（B = o·d > 0）近乎切线。
    # 朴素正根 -B + sqrt(B^2 - c) 会发生相消；稳定形式 -c/(B + sqrt(...))
    # 应给出与解析解一致的正根。
    R = 10.0

    # 起点 (R-δ, 0, 0)，方向向北水平（B = 0）：精确根 sqrt(R^2 - (R-δ)^2)。
    # 该情形根由 |o| 与 R 的浮点表示之差决定（固有的 ~eps·R/δ 量级误差），
    # 不检验抵消；只需相对误差远小于 1 且交点严格在球面上。
    delta = 1e-7
    ox = R - delta
    t_exact = math.sqrt(2.0 * R * delta - delta * delta)
    t = ray_sphere_intersect(Vec3(ox, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), R)
    assert t == pytest.approx(t_exact, rel=1e-7)
    assert t > 0.0
    assert ox ** 2 + t ** 2 == pytest.approx(R ** 2, abs=1e-8)

    # 朝几乎直指出口方向（方向与 o 同向，向北偏 1e-9 弧度），δ 仅 1e-12：
    # 正根应是约 2δ 的极小正数（朴素 -B + sqrt(B^2 - c) 在此尺度严重相消；
    # 因式分解的 c 与稳定商式保证结果仍落在球面上）。
    delta2 = 1e-12
    ox2 = R - delta2
    small_ang = 1e-9  # 方向相对径向（+E）向北偏 1e-9 弧度
    d = Vec3(math.cos(small_ang), math.sin(small_ang), 0.0)
    t2 = ray_sphere_intersect(Vec3(ox2, 0.0, 0.0), d, R)
    assert 0.0 < t2 < 1e-8
    # 交点严格落在球面上（近切线精度的最终判据）
    pe = ox2 + t2 * d.e
    pn = t2 * d.n
    assert pe ** 2 + pn ** 2 == pytest.approx(R ** 2, abs=1e-10)

    # B < 0 分支：起点贴近西侧内壁 (-R+δ, 0, 0)，射线指向东（+E，背离近侧
    # 出口、奔向远侧内壁）：B = o·d < 0，正根为约 2R-δ 的长根。
    o3 = Vec3(-(R - 1e-6), 0.0, 0.0)
    t3 = ray_sphere_intersect(o3, Vec3(1.0, 0.0, 0.0), R)
    assert t3 == pytest.approx(2.0 * R - 1e-6, rel=1e-9)

    # 带 b>0 的近切线（朝接近“出口”方向），同样验证稳定根
    # 起点 (ox,0,0)，方向略偏东，仍近似擦过球面前行
    ang = math.radians(1e-7)
    d = Vec3(math.sin(ang), math.cos(ang), 0.0)
    hit = project_ray(Vec3(ox, 0.0, 0.0), math.degrees(ang), 0.0, R)
    assert hit.point.e ** 2 + hit.point.n ** 2 == pytest.approx(R ** 2, abs=1e-7)
    assert hit.distance > 0.0
    assert d.e * hit.distance + ox <= R + 1e-6


def test_origin_outside_or_on_sphere_rejected():
    with pytest.raises(ApertureValidationError) as ei:
        ensure_origin_inside(Vec3(10.0001, 0.0, 0.0), 10.0)
    assert ei.value.payload["reason"] == "origin_outside_sphere"
    # 恰在球面上同样拒绝（要求严格球内，前向射线可能无正根）
    with pytest.raises(ApertureValidationError):
        ensure_origin_inside(Vec3(0.0, 6.0, 8.0), 10.0)


# ---------------------------------------------------------------------------
# 最短覆盖圆弧 / 包络
# ---------------------------------------------------------------------------


def test_covering_arc_plain():
    center, width, crosses = shortest_covering_arc([10.0, 20.0, 30.0])
    assert (center, width, crosses) == (20.0, 20.0, False)


def test_covering_arc_single_point():
    assert shortest_covering_arc([45.0, 45.0]) == (45.0, 0.0, False)


def test_covering_arc_crosses_zero():
    # 点 350/355/0/5/10：覆盖弧应跨 0°，center=0，width=20
    center, width, crosses = shortest_covering_arc(
        [350.0, 355.0, 0.0, 5.0, 10.0]
    )
    assert crosses is True
    assert center == pytest.approx(0.0, abs=1e-9)
    assert width == pytest.approx(20.0, abs=1e-9)


def test_covering_arc_tie_three_equal_gaps_picks_smaller_center():
    # {0, 120, 240}：三个间隙均为 120°，三个候选中心为 240/0/120，
    # 并列取 center 较小者 → 0，覆盖弧 [120→360] 跨 0°，宽 240。
    center, width, crosses = shortest_covering_arc([0.0, 120.0, 240.0])
    assert center == pytest.approx(0.0, abs=1e-12)
    assert width == pytest.approx(240.0, abs=1e-12)
    assert crosses is True


def test_covering_arc_tie_two_gaps_picks_smaller_center():
    # 对径两点 {0, 180}：两候选中心 270（跨 0）与 90（不跨），取较小 90。
    center, width, crosses = shortest_covering_arc([0.0, 180.0])
    assert center == pytest.approx(90.0, abs=1e-12)
    assert width == pytest.approx(180.0, abs=1e-12)
    assert crosses is False

    # {90, 270}：候选中心 0（跨 0）与 180（不跨），取较小 0。
    center, width, crosses = shortest_covering_arc([90.0, 270.0])
    assert center == pytest.approx(0.0, abs=1e-12)
    assert crosses is True


def test_envelope_uses_spherical_angles_of_intersections():
    origin = Vec3(1.5, 0.0, 1.0)
    hits = [project_ray(origin, az, el, 10.0)
            for az, el in [(10.0, 20.0), (12.0, 18.0), (8.0, 22.0),
                           (12.0, 22.0), (8.0, 18.0)]]
    env = aperture_envelope(hits)
    azs = [h.azimuth for h in hits]
    els = [h.elevation for h in hits]
    assert env.elevation_low == pytest.approx(min(els), abs=1e-12)
    assert env.elevation_high == pytest.approx(max(els), abs=1e-12)
    # 所有点都在包络圆弧内（角距中心 <= 半宽）
    for a in azs:
        d = abs((a - env.azimuth_center + 180.0) % 360.0 - 180.0)
        assert d <= env.azimuth_width / 2.0 + 1e-7


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_api_nominal_centric():
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 0.0), radius=10.0,
                      samples=[_sample(0, 30.0, 45.0, daz=2.0, dele=1.0)]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 1
    res = body["results"][0]
    nom = res["nominal"]
    assert nom["distance"] == pytest.approx(10.0, abs=1e-8)
    # 同心球：交点方位/高度即视线方位/高度
    assert nom["azimuth"] == pytest.approx(30.0, abs=1e-8)
    assert nom["elevation"] == pytest.approx(45.0, abs=1e-8)
    assert len(res["corners"]) == 4
    assert [c["daz"] for c in res["corners"]] == [-2.0, 2.0, -2.0, 2.0]
    assert [c["dele"] for c in res["corners"]] == [-1.0, -1.0, 1.0, 1.0]
    env = res["envelope"]
    assert env["azimuth_center"] == pytest.approx(30.0, abs=1e-8)
    assert env["azimuth_width"] == pytest.approx(4.0, abs=1e-8)
    assert env["azimuth_crosses_zero"] is False
    assert env["elevation_low"] == pytest.approx(44.0, abs=1e-8)
    assert env["elevation_high"] == pytest.approx(46.0, abs=1e-8)


def test_api_eccentric():
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(2.0, -1.0, 3.0), radius=12.0,
                      samples=[_sample(0, 200.0, 15.0, daz=1.0, dele=0.5),
                               _sample(1, 0.0, 80.0)]),
    )
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1]
    for res in results:
        p = res["nominal"]["point"]
        assert p["east"] ** 2 + p["north"] ** 2 + p["up"] ** 2 == \
            pytest.approx(144.0, abs=1e-6)
        assert res["nominal"]["distance"] > 0.0
        for c in res["corners"]:
            cp = c["intersection"]["point"]
            assert cp["east"] ** 2 + cp["north"] ** 2 + cp["up"] ** 2 == \
                pytest.approx(144.0, abs=1e-6)


def test_api_envelope_crosses_zero():
    # az=0、δaz=10、圆心时四角方位 ±10 → 包络跨 0°，center=0、width=20
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(samples=[_sample(0, 0.0, 30.0, daz=10.0, dele=2.0)]),
    )
    assert r.status_code == 200, r.text
    env = r.json()["results"][0]["envelope"]
    assert env["azimuth_crosses_zero"] is True
    assert env["azimuth_center"] == pytest.approx(0.0, abs=1e-8)
    assert env["azimuth_width"] == pytest.approx(20.0, abs=1e-8)
    assert env["elevation_low"] == pytest.approx(28.0, abs=1e-8)
    assert env["elevation_high"] == pytest.approx(32.0, abs=1e-8)


def test_api_near_tangent_regression():
    # 起点距球面仅 1e-6，近切线射线；稳定求根应给出正距离且交点严格在球面上。
    # 标称向北水平（B=0）时正根 ≈ sqrt(2R·δ) ≈ 4.4721e-3；高度取 10° 使
    # ±0.001° 的误差角交点也都在穹顶地平面以上。
    delta = 1e-6
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(10.0 - delta, 0.0, 0.0), radius=10.0,
                      samples=[_sample(0, 0.0, 10.0, daz=0.001, dele=0.001)]),
    )
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["nominal"]["distance"] > 0.0
    for hit in [res["nominal"]] + [c["intersection"] for c in res["corners"]]:
        p = hit["point"]
        assert p["east"] ** 2 + p["north"] ** 2 + p["up"] ** 2 == \
            pytest.approx(100.0, abs=1e-7)
        assert hit["distance"] > 0.0
        assert p["up"] > 0.0


def test_api_origin_outside_sphere_422():
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(10.0, 0.0, 0.0), radius=10.0),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["reason"] == "origin_outside_sphere"
    assert body["dome_radius"] == 10.0
    assert "origin" in body

    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(6.0, 8.0, 0.0), radius=10.0),  # 恰在球面上
    )
    assert r.status_code == 422


def test_api_elevation_out_of_range_reports_indices():
    # 样本 0 合法；样本 1 标称高度越界；样本 2 角点高度 el+δel 越界
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(samples=[
            _sample(0, 0.0, 10.0),
            _sample(1, 0.0, 91.0),
            _sample(2, 0.0, 89.5, dele=1.0),
            _sample(3, 0.0, -89.5, dele=1.0),  # -90.5 越界
        ]),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["elevation_out_of_range_indices"] == [1, 2, 3]
    assert body["below_ground_indices"] == []


def test_api_below_ground_reports_indices():
    # 起点 (0,0,1)，R=10：el=0 时交点 u=1 在地面之上（合法）；
    # 视线陡向下（el=-30°）使交点 u<0（低于地平面）→ 样本非法并指出索引。
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 1.0), radius=10.0, samples=[
            _sample(0, 0.0, 0.0),             # u=1 > 0 合法
            _sample(1, 0.0, -30.0, dele=0.0),  # 向下倾斜，交点 u<0
        ]),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["below_ground_indices"] == [1]
    assert body["elevation_out_of_range_indices"] == []


def test_api_both_kinds_of_bad_samples_collected():
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 1.0), radius=10.0, samples=[
            _sample(0, 0.0, 95.0),            # 高度越界
            _sample(1, 0.0, -30.0),           # 低于地平面
            _sample(2, 0.0, 10.0),            # 合法
        ]),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["elevation_out_of_range_indices"] == [0]
    assert body["below_ground_indices"] == [1]


def test_api_origin_below_ground_but_inside_is_acceptable():
    # 起点位于地平面以下但仍在球内：几何上允许；射线向上仍可命中穹顶内表面
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, -1.0), radius=10.0,
                      samples=[_sample(0, 0.0, 60.0)]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["nominal"]["point"]["up"] > 0.0


def test_api_boundary_elevation_and_ground_plane_are_valid():
    # 标称高度恰为 +90°（天顶）、角点高度恰为 +90°：高度角上边界本身合法。
    # el=-90 自任意球内点直指天底，交点恒在地面以下，由地平面规则拒绝。
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 0.0), radius=10.0, samples=[
            _sample(0, 0.0, 90.0),
            _sample(1, 0.0, 89.0, dele=1.0),  # 角点恰 90
        ]),
    )
    assert r.status_code == 200, r.text

    # 高度角恰 -90° 不越出角度范围（不应出现在 elevation_out_of_range_indices），
    # 但其交点在地面以下 → 出现在 below_ground_indices
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 0.0), radius=10.0,
                      samples=[_sample(0, 0.0, -90.0)]),
    )
    assert r.status_code == 422
    body = r.json()
    assert body["elevation_out_of_range_indices"] == []
    assert body["below_ground_indices"] == [0]

    # 交点恰好落在地平面（圆心起点、el=0 → up=0）：合法（仅严格低于才拒绝）
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 0.0, 0.0), radius=10.0,
                      samples=[_sample(0, 0.0, 0.0)]),
    )
    assert r.status_code == 200, r.text
    p = r.json()["results"][0]["nominal"]["point"]
    assert p["up"] == pytest.approx(0.0, abs=1e-8)


def test_api_envelope_eccentric_arc_contains_all_points():
    # 偏心起点：交点的球面方位相对视线方位发生偏移，包络必须按球面方位
    # （而非输入视线方位）构造，且五个交点都落在包络圆弧与高度界内。
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(2.0, 0.5, 1.0), radius=10.0,
                      samples=[_sample(0, 0.0, 45.0, daz=5.0, dele=3.0)]),
    )
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    env = res["envelope"]
    assert env["azimuth_crosses_zero"] is False
    hits = [res["nominal"]] + [c["intersection"] for c in res["corners"]]
    for h in hits:
        d = abs((h["azimuth"] - env["azimuth_center"] + 180.0) % 360.0 - 180.0)
        assert d <= env["azimuth_width"] / 2.0 + 1e-7
    els = [h["elevation"] for h in hits]
    assert env["elevation_low"] == pytest.approx(min(els), abs=1e-8)
    assert env["elevation_high"] == pytest.approx(max(els), abs=1e-8)


def test_api_envelope_eccentric_crossing_zero():
    # 偏心起点同时误差矩形跨越北方向：包络仍应跨 0°
    r = client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(origin=(0.0, 3.0, 1.0), radius=10.0,
                      samples=[_sample(0, 0.0, 50.0, daz=6.0, dele=2.0)]),
    )
    assert r.status_code == 200, r.text
    env = r.json()["results"][0]["envelope"]
    assert env["azimuth_crosses_zero"] is True
    assert 0.0 <= env["azimuth_center"] < 360.0
    assert 0.0 < env["azimuth_width"] < 360.0


def test_api_index_must_match_position():
    bad = _payload(samples=[_sample(1, 0.0, 10.0)])
    r = client.post("/api/v1/project-dome-aperture", json=bad)
    assert r.status_code == 422


def test_api_validation_bad_radius_and_empty_samples():
    assert client.post(
        "/api/v1/project-dome-aperture",
        json=_payload() | {"dome_radius": 0.0},
    ).status_code == 422
    assert client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(samples=[]),
    ).status_code == 422
    assert client.post(
        "/api/v1/project-dome-aperture",
        json=_payload(samples=[_sample(0, 0.0, 10.0, daz=-1.0)]),
    ).status_code == 422
