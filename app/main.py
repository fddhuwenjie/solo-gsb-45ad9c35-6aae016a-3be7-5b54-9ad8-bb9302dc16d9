"""FastAPI 入口：健康检查、圆顶狭缝跟踪求解与安全联锁审计接口。"""

from __future__ import annotations

import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models import (
    ActionItem,
    ApertureCorner,
    ApertureEnvelope,
    ApertureIntersection,
    ApertureSampleResult,
    AuditSafetyInterlockRequest,
    AuditSafetyInterlockResponse,
    CalibrateSyncRequest,
    CalibrateSyncResponse,
    EnuPoint,
    FailureDetail,
    HealthResponse,
    InterlockCommandRecord,
    InterlockRiskInterval,
    InterlockShutterSegment,
    InterlockStatusSegment,
    ProjectDomeApertureRequest,
    ProjectDomeApertureResponse,
    PlanDomeBalanceRequest,
    PlanDomeBalanceResponse,
    BalanceWeightItem,
    MomentVector2,
    PlanWeightSequenceRequest,
    PlanWeightSequenceResponse,
    WeightSequenceState,
    WeightSequenceStep,
    BlockedWeightAction,
    TrackRequest,
    TrackResponse,
    WorstDeviationItem,
)
from .planner.aperture import (
    ANGLE_EPS,
    ApertureValidationError,
    Vec3,
    aperture_envelope,
    ensure_origin_inside,
    project_ray,
)
from .planner.balance import Slot, plan_balance, slot_levels
from .planner.calibration import calibrate_sync, normalize_bias
from .planner.interlock import Event, InterlockValidationError, audit_interlock
from .planner.resequence import (
    WeightSequenceValidationError,
    plan_weight_sequence,
)
from .planner.solver import Problem, solve

app = FastAPI(
    title="天文台圆顶狭缝跟踪 API",
    description=(
        "为时间序列目标方位计算圆顶动作时间线：目标始终位于狭缝内、"
        "满足转速与线缆边界约束、累计转角最小，等代价取字典序最小序列。"
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="dome-slit-tracker")


@app.exception_handler(InterlockValidationError)
async def interlock_validation_error_handler(
    _request: Request, exc: InterlockValidationError
) -> JSONResponse:
    # 事件流语义非法（快门缺初值 / 跳态等）→ HTTP 422
    return JSONResponse(
        status_code=422,
        content={"detail": exc.message, "event_index": exc.event_index},
    )


@app.exception_handler(ApertureValidationError)
async def aperture_validation_error_handler(
    _request: Request, exc: ApertureValidationError
) -> JSONResponse:
    # 视轴起点不在球内 / 误差越出高度角范围 / 交点低于地平面 → HTTP 422
    return JSONResponse(status_code=422, content=exc.payload)


@app.exception_handler(WeightSequenceValidationError)
async def weight_sequence_validation_error_handler(
    _request: Request, exc: WeightSequenceValidationError
) -> JSONResponse:
    # 起点或终点状态本身越限（总质量/残余质量矩）→ HTTP 422
    return JSONResponse(status_code=422, content=exc.payload)


@app.post("/api/v1/track", response_model=TrackResponse, tags=["tracking"])
def track(req: TrackRequest) -> TrackResponse:
    problem = Problem(
        target_azimuths=req.target_azimuths,
        times=req.times,
        initial_angle=req.initial_angle,
        max_speed=req.max_speed,
        slit_width=req.slit_width,
        cable_lo=req.cable_low,
        cable_hi=req.cable_high,
    )
    sol = solve(problem)

    if sol.feasible:
        return TrackResponse(
            feasible=True,
            total_rotation=round(sol.total_rotation, 9),
            actions=[
                ActionItem(
                    index=a.index,
                    time=a.time,
                    dome_angle=round(a.dome_angle, 9),
                    target_azimuth=a.target_azimuth,
                    lift=a.lift,
                    move_delta=round(a.move_delta, 9),
                    move_speed=round(a.move_speed, 9),
                    cable_low=a.cable_low,
                    cable_high=a.cable_high,
                )
                for a in sol.actions
            ],
        )

    f = sol.failure
    return TrackResponse(
        feasible=False,
        failure=FailureDetail(
            index=f.index,
            time=f.time,
            target_azimuth=None if (isinstance(f.target_azimuth, float) and
                                    math.isnan(f.target_azimuth)) else f.target_azimuth,
            unmet_constraints=f.unmet,  # type: ignore[arg-type]
            details=f.details,
        ),
    )


@app.post(
    "/api/v1/calibrate-sync",
    response_model=CalibrateSyncResponse,
    tags=["calibration"],
)
def calibrate(req: CalibrateSyncRequest) -> CalibrateSyncResponse:
    outcome = calibrate_sync(
        reference_times=req.reference_times,
        reference_angles=req.reference_angles,
        encoder_times=req.encoder_times,
        encoder_angles=req.encoder_angles,
        lag_min=req.lag_min,
        lag_max=req.lag_max,
        lag_step=req.lag_step,
        outlier_threshold=req.outlier_threshold,
        min_inliers=req.min_inliers,
    )

    if not outcome.calibrated:
        return CalibrateSyncResponse(
            calibrated=False,
            reason=outcome.reason,
            details=outcome.details,
        )

    return CalibrateSyncResponse(
        calibrated=True,
        bias_deg=round(normalize_bias(outcome.bias), 9),
        lag_seconds=round(outcome.lag, 9),
        rmse_deg=round(outcome.rmse, 9),
        inlier_indices=outcome.inliers,
        outlier_indices=outcome.outliers,
        details=outcome.details,
    )


def _r9(value: float | None) -> float | None:
    return None if value is None else round(value, 9)


@app.post(
    "/api/v1/audit-safety-interlock",
    response_model=AuditSafetyInterlockResponse,
    tags=["interlock"],
)
def audit_safety_interlock(
    req: AuditSafetyInterlockRequest,
) -> AuditSafetyInterlockResponse:
    events = [
        Event(
            index=i,
            time=e.time,
            kind=e.kind,
            value=getattr(e, "value", None),
            state=getattr(e, "state", None),
        )
        for i, e in enumerate(req.events)
    ]
    result = audit_interlock(
        events,
        wind_trip=req.wind_trip,
        wind_reset=req.wind_reset,
        rain_trip=req.rain_trip,
        rain_reset=req.rain_reset,
        trip_duration=req.trip_duration,
        reset_duration=req.reset_duration,
        stale_after=req.stale_after,
        command_deadline=req.command_deadline,
        close_deadline=req.close_deadline,
    )

    return AuditSafetyInterlockResponse(
        status_timeline=[
            InterlockStatusSegment(
                start=round(s.start, 9),
                end=_r9(s.end),
                state=s.state,  # type: ignore[arg-type]
                cause_event_indices=s.cause_event_indices,
                projected=s.projected,
            )
            for s in result.status_timeline
        ],
        risk_intervals=[
            InterlockRiskInterval(
                start=round(r.start, 9),
                end=_r9(r.end),
                states=r.states,  # type: ignore[arg-type]
                entry_event_indices=r.entry_event_indices,
                projected=r.projected,
                command_due=round(r.command_due, 9),
                close_due=round(r.close_due, 9),
                command_time=_r9(r.command_time),
                command_event_index=r.command_event_index,
                closed_time=_r9(r.closed_time),
                closed_event_index=r.closed_event_index,
                already_closed_at_entry=r.already_closed_at_entry,
                command_overdue=round(r.command_overdue, 9),
                close_overdue=round(r.close_overdue, 9),
            )
            for r in result.risk_intervals
        ],
        close_commands=[
            InterlockCommandRecord(
                time=round(c.time, 9), event_index=c.event_index
            )
            for c in result.close_commands
        ],
        shutter_timeline=[
            InterlockShutterSegment(
                start=round(s.start, 9),
                end=_r9(s.end),
                state=s.state,  # type: ignore[arg-type]
                event_index=s.event_index,
            )
            for s in result.shutter_timeline
        ],
        horizon=round(result.horizon, 9),
    )


# 误差矩形四角的 (方位偏移, 高度偏移) 固定顺序
_APERTURE_CORNER_SIGNS = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, 1.0),
)


def _aperture_intersection_model(hit) -> ApertureIntersection:
    return ApertureIntersection(
        distance=round(hit.distance, 9),
        point=EnuPoint(
            east=round(hit.point.e, 9),
            north=round(hit.point.n, 9),
            up=round(hit.point.u, 9),
        ),
        azimuth=round(hit.azimuth, 9),
        elevation=round(hit.elevation, 9),
    )


@app.post(
    "/api/v1/project-dome-aperture",
    response_model=ProjectDomeApertureResponse,
    tags=["aperture"],
)
def project_dome_aperture(
    req: ProjectDomeApertureRequest,
) -> ProjectDomeApertureResponse:
    radius = req.dome_radius
    origin = Vec3(req.origin.east, req.origin.north, req.origin.up)

    # 视轴起点必须严格位于球内（含球面上也拒绝）
    ensure_origin_inside(origin, radius)

    out_of_range_indices: list[int] = []
    below_ground_indices: list[int] = []
    sample_results: list[ApertureSampleResult] = []

    ground_tol = 1e-9 * radius

    for s in req.samples:
        corner_angles = [
            (s.azimuth + sa * s.azimuth_error,
             s.elevation + se * s.elevation_error)
            for sa, se in _APERTURE_CORNER_SIGNS
        ]

        # 标称及四角高度角都必须落在 [-90, 90]
        el_values = [s.elevation] + [el for _az, el in corner_angles]
        if not all(-90.0 - ANGLE_EPS <= el <= 90.0 + ANGLE_EPS
                   for el in el_values):
            out_of_range_indices.append(s.index)
            continue

        hits = [project_ray(origin, s.azimuth, s.elevation, radius)]
        hits.extend(
            project_ray(origin, az, el, radius) for az, el in corner_angles
        )

        # 任一交点低于穹顶地平面（U < 0）即非法
        if any(h.point.u < -ground_tol for h in hits):
            below_ground_indices.append(s.index)
            continue

        env = aperture_envelope(hits)
        sample_results.append(
            ApertureSampleResult(
                index=s.index,
                nominal=_aperture_intersection_model(hits[0]),
                corners=[
                    ApertureCorner(
                        daz=round(sa * s.azimuth_error, 9),
                        dele=round(se * s.elevation_error, 9),
                        intersection=_aperture_intersection_model(hits[k + 1]),
                    )
                    for k, (sa, se) in enumerate(_APERTURE_CORNER_SIGNS)
                ],
                envelope=ApertureEnvelope(
                    azimuth_center=round(env.azimuth_center, 9),
                    azimuth_width=round(env.azimuth_width, 9),
                    azimuth_crosses_zero=env.azimuth_crosses_zero,
                    elevation_low=round(env.elevation_low, 9),
                    elevation_high=round(env.elevation_high, 9),
                ),
            )
        )

    if out_of_range_indices or below_ground_indices:
        raise ApertureValidationError({
            "detail": (
                "存在非法样本：误差角越出高度角范围 [-90, 90]，"
                "或交点低于穹顶地平面（详见样本索引列表）"
            ),
            "elevation_out_of_range_indices": out_of_range_indices,
            "below_ground_indices": below_ground_indices,
        })

    return ProjectDomeApertureResponse(
        dome_radius=radius,
        origin=EnuPoint(east=origin.e, north=origin.n, up=origin.u),
        results=sample_results,
    )


@app.post(
    "/api/v1/plan-dome-balance",
    response_model=PlanDomeBalanceResponse,
    tags=["balance"],
)
def plan_dome_balance(
    req: PlanDomeBalanceRequest,
) -> PlanDomeBalanceResponse:
    # 规划器内部按槽位 ID 升序枚举，请求数组顺序不影响结果。
    slots = [
        Slot(
            id=s.id,
            radius=s.radius,
            azimuth_deg=s.azimuth,
            mass_limit=s.mass_limit,
            levels=slot_levels(s.mass_limit, req.mass_step),
        )
        for s in sorted(req.slots, key=lambda s: s.id)
    ]
    plan = plan_balance(
        parts=[(p.mass, p.radius, p.azimuth) for p in req.installed_parts],
        slots=slots,
        mass_step=req.mass_step,
        mass_tolerance_ratio=req.mass_tolerance_ratio,
        azimuth_tolerance_deg=req.azimuth_tolerance,
        residual_moment_limit=req.residual_moment_limit,
    )

    return PlanDomeBalanceResponse(
        original_moment=MomentVector2(
            east=round(plan.original_x, 9),
            north=round(plan.original_y, 9),
        ),
        original_magnitude=round(plan.original_magnitude, 9),
        weights=[
            BalanceWeightItem(
                slot_id=w.slot_id,
                radius=round(w.radius, 9),
                nominal_mass=round(w.nominal_mass, 9),
                nominal_azimuth=round(w.nominal_azimuth_deg, 9),
                worst_mass=round(w.worst_mass, 9),
                worst_azimuth=round(w.worst_azimuth_deg, 9),
            )
            for w in plan.weights
        ],
        total_balance_mass=round(plan.total_balance_mass, 9),
        nominal_residual=MomentVector2(
            east=round(plan.nominal_x, 9),
            north=round(plan.nominal_y, 9),
        ),
        nominal_magnitude=round(plan.nominal_magnitude, 9),
        worst_residual=MomentVector2(
            east=round(plan.worst_x, 9),
            north=round(plan.worst_y, 9),
        ),
        worst_magnitude=round(plan.worst_magnitude, 9),
        worst_deviation_combo=[
            WorstDeviationItem(
                slot_id=c.slot_id,
                mass=round(c.mass, 9),
                azimuth=round(c.azimuth_deg, 9),
                mass_sign=c.mass_sign,
                azimuth_sign=c.azimuth_sign,
            )
            for c in plan.worst_combo
        ],
        residual_moment_limit=round(plan.residual_limit, 9),
        compliant=plan.compliant,
        details={
            "evaluated_plans": plan.evaluated_plans,
            "max_scenarios_per_plan": plan.max_scenarios_per_plan,
            "slot_ids": list(plan.slot_ids),
            "mass_vector": [round(m, 9) for m in plan.mass_vector],
        },
    )


def _weight_sequence_state_model(snap) -> WeightSequenceState:
    return WeightSequenceState(
        installed_ids=list(snap.installed_ids),
        total_mass=round(snap.total_mass, 9),
        residual=MomentVector2(
            east=round(snap.residual_east, 9),
            north=round(snap.residual_north, 9),
        ),
        residual_magnitude=round(snap.residual_magnitude, 9),
    )


@app.post(
    "/api/v1/plan-weight-sequence",
    response_model=PlanWeightSequenceResponse,
    tags=["balance"],
)
def plan_weight_sequence_endpoint(
    req: PlanWeightSequenceRequest,
) -> PlanWeightSequenceResponse:
    plan = plan_weight_sequence(
        fixed_parts=[(p.mass, p.radius, p.azimuth) for p in req.fixed_parts],
        weight_items=[(w.id, w.mass, w.radius, w.azimuth) for w in req.weights],
        current_ids=set(req.current_ids),
        target_ids=set(req.target_ids),
        installed_mass_limit=req.installed_mass_limit,
        residual_moment_limit=req.residual_moment_limit,
    )

    blocked = None
    if plan.nearest_blocked_action is not None:
        b = plan.nearest_blocked_action
        blocked = BlockedWeightAction(
            action=b.action,
            weight_id=b.weight_id,
            from_depth=b.from_depth,
            from_installed_ids=list(b.from_installed_ids),
            resulting_installed_ids=list(b.resulting_installed_ids),
            resulting_total_mass=round(b.resulting_total_mass, 9),
            resulting_residual=MomentVector2(
                east=round(b.resulting_residual_east, 9),
                north=round(b.resulting_residual_north, 9),
            ),
            resulting_residual_magnitude=round(b.resulting_residual_magnitude, 9),
            moment_excess=round(b.moment_excess, 9),
            mass_excess=round(b.mass_excess, 9),
        )

    return PlanWeightSequenceResponse(
        feasible=plan.feasible,
        fixed_total_mass=round(plan.fixed_mass, 9),
        fixed_moment=MomentVector2(
            east=round(plan.fixed_moment_x, 9),
            north=round(plan.fixed_moment_y, 9),
        ),
        fixed_moment_magnitude=round(plan.fixed_moment_magnitude, 9),
        installed_mass_limit=round(plan.installed_mass_limit, 9),
        residual_moment_limit=round(plan.residual_moment_limit, 9),
        initial_state=_weight_sequence_state_model(plan.initial),
        target_state=_weight_sequence_state_model(plan.target),
        target_ids=list(plan.target_ids),
        steps=[
            WeightSequenceStep(
                index=s.index,
                action=s.action,
                weight_id=s.weight_id,
                installed_ids=list(s.installed_ids),
                total_mass=round(s.total_mass, 9),
                residual=MomentVector2(
                    east=round(s.residual_east, 9),
                    north=round(s.residual_north, 9),
                ),
                residual_magnitude=round(s.residual_magnitude, 9),
                temporary=s.temporary,
            )
            for s in plan.steps
        ],
        temporary_weight_ids=list(plan.temporary_weight_ids),
        reachable_states=plan.reachable_states,
        max_intermediate_magnitude=round(plan.max_intermediate_magnitude, 9),
        nearest_blocked_action=blocked,
        details={"explored_states": plan.explored_states},
    )
