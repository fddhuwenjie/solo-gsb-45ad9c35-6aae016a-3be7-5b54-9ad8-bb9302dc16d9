"""请求/响应数据模型。角度单位为度，时间单位为秒。"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .planner.balance import (
    MAX_LEVELS_PER_SLOT as BALANCE_MAX_LEVELS_PER_SLOT,
)
from .planner.balance import MAX_SLOTS as BALANCE_MAX_SLOTS
from .planner.resequence import MAX_WEIGHTS as RESEQUENCE_MAX_WEIGHTS
from .planner.calibration import (
    MAX_LAG_CANDIDATES,
    unwrap_shortest_arc,
)


class TrackRequest(BaseModel):
    target_azimuths: list[float] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="各时刻目标方位角（度），长度需与 times 相同",
    )
    times: list[float] = Field(
        ...,
        min_length=1,
        description="各目标对应的绝对时刻（秒，单调严格递增，首个应 >= 0）",
    )
    initial_angle: float = Field(
        ..., description="圆顶当前未展开角度（度，可取任意实数）"
    )
    max_speed: float = Field(
        ..., gt=0, description="圆顶最大转速（度/秒，必须为正）"
    )
    slit_width: float = Field(
        ..., gt=0, le=360, description="狭缝角宽（度，(0, 360]）"
    )
    cable_low: float = Field(..., description="线缆角度下界（未展开角，度）")
    cable_high: float = Field(..., description="线缆角度上界（未展开角，度）")

    @model_validator(mode="after")
    def _validate(self) -> "TrackRequest":
        if len(self.target_azimuths) != len(self.times):
            raise ValueError("target_azimuths 与 times 长度必须相同")
        if self.cable_low > self.cable_high:
            raise ValueError("cable_low 必须小于等于 cable_high")
        if any(t <= 0 for t in self.times):
            raise ValueError("times 必须为正数（首个目标时刻 t1 > 0）")
        for a, b in zip(self.times, self.times[1:]):
            if b <= a:
                raise ValueError("times 必须严格递增")
        if self.initial_angle < self.cable_low or self.initial_angle > self.cable_high:
            raise ValueError("initial_angle 必须位于线缆区间 [cable_low, cable_high] 内")
        return self


class ActionItem(BaseModel):
    index: int
    time: float
    dome_angle: float = Field(..., description="该时刻选定的圆顶未展开角度（度）")
    target_azimuth: float
    lift: int = Field(..., description="目标所在的整数圈（目标 = azimuth + 360*lift）")
    move_delta: float = Field(..., description="相对上一时刻的转角（度，带符号）")
    move_speed: float = Field(..., description="本步实际转速（度/秒，带符号）")
    cable_low: float
    cable_high: float


class FailureDetail(BaseModel):
    index: int = Field(..., description="最早失效时刻序号（1 基；0 表示初始状态即不合法）")
    time: float
    target_azimuth: float | None
    unmet_constraints: list[Literal["slit_visibility", "speed_limit", "cable_limit"]]
    details: dict


class TrackResponse(BaseModel):
    feasible: bool
    total_rotation: float = Field(0.0, description="累计转角（度）= Σ|move_delta|")
    actions: list[ActionItem] = Field(default_factory=list)
    failure: FailureDetail | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


def _strictly_increasing(values: list[float], name: str) -> None:
    for a, b in zip(values, values[1:]):
        if b <= a:
            raise ValueError(f"{name} 必须严格递增")


def _all_finite(values: list[float], name: str) -> None:
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{name} 不得包含 NaN 或无穷大")


class CalibrateSyncRequest(BaseModel):
    reference_times: list[float] = Field(
        ...,
        min_length=2,
        description="参考方位样本的绝对时刻（秒，严格递增）",
    )
    reference_angles: list[float] = Field(
        ...,
        min_length=2,
        description="各参考时刻的参考方位角（度，与 reference_times 等长）",
    )
    encoder_times: list[float] = Field(
        ...,
        min_length=2,
        description="圆顶编码器样本的绝对时刻（秒，严格递增）",
    )
    encoder_angles: list[float] = Field(
        ...,
        min_length=2,
        description="各编码器时刻的编码器方位角（度，与 encoder_times 等长）",
    )
    lag_min: float = Field(..., description="时延搜索下界（秒）")
    lag_max: float = Field(..., description="时延搜索上界（秒）")
    lag_step: float = Field(..., gt=0, description="时延枚举步长（秒，> 0）")
    outlier_threshold: float = Field(
        ..., gt=0, description="离群点阈值（度，残差偏离中位数初估值超过它即剔除）"
    )
    min_inliers: int = Field(
        ..., ge=1, description="有效候选所需的最少内点数（>= 1）"
    )

    @model_validator(mode="after")
    def _validate(self) -> "CalibrateSyncRequest":
        if len(self.reference_times) != len(self.reference_angles):
            raise ValueError("reference_times 与 reference_angles 长度必须相同")
        if len(self.encoder_times) != len(self.encoder_angles):
            raise ValueError("encoder_times 与 encoder_angles 长度必须相同")

        _all_finite(self.reference_times, "reference_times")
        _all_finite(self.reference_angles, "reference_angles")
        _all_finite(self.encoder_times, "encoder_times")
        _all_finite(self.encoder_angles, "encoder_angles")
        if not all(math.isfinite(v) for v in (self.lag_min, self.lag_max,
                                              self.lag_step, self.outlier_threshold)):
            raise ValueError("lag_min/lag_max/lag_step/outlier_threshold 必须为有限数")

        _strictly_increasing(self.reference_times, "reference_times")
        _strictly_increasing(self.encoder_times, "encoder_times")

        # 相邻角度按最短弧展开；恰为 180° 的跳变方向有歧义，直接拒绝。
        unwrap_shortest_arc(self.reference_angles, "reference_angles")
        unwrap_shortest_arc(self.encoder_angles, "encoder_angles")

        if self.lag_min > self.lag_max:
            raise ValueError("lag_min 必须小于等于 lag_max")
        if self.lag_step <= 0:
            raise ValueError("lag_step 必须为正数")
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold 必须为正数")
        if self.min_inliers < 1:
            raise ValueError("min_inliers 必须 >= 1")
        n_candidates = math.floor((self.lag_max - self.lag_min) / self.lag_step) + 1
        if n_candidates > MAX_LAG_CANDIDATES:
            raise ValueError(
                f"时延候选数 {n_candidates} 超过上限 {MAX_LAG_CANDIDATES}"
            )
        return self


class CalibrateSyncResponse(BaseModel):
    calibrated: bool = Field(..., description="是否存在有效候选")
    bias_deg: float = Field(
        0.0, description="零位偏差（度，归一化到 [-180, 180)，encoder + bias ≈ reference）"
    )
    lag_seconds: float = Field(0.0, description="响应时延（秒，正值表示编码器滞后）")
    rmse_deg: float = Field(0.0, description="内点残差 RMSE（度）")
    inlier_indices: list[int] = Field(
        default_factory=list,
        description="内点在编码器样本中的 0 基原始索引（升序）",
    )
    outlier_indices: list[int] = Field(
        default_factory=list,
        description="离群点在编码器样本中的 0 基原始索引（升序）",
    )
    reason: str | None = Field(None, description="无有效候选时的原因说明")
    details: dict = Field(
        default_factory=dict, description="枚举候选数、最大重叠/内点数等诊断信息"
    )


# ---------------------------------------------------------------------------
# POST /api/v1/audit-safety-interlock
# ---------------------------------------------------------------------------

WeatherState = Literal["safe", "unsafe", "unknown"]
ShutterState = Literal["open", "closing", "closed", "opening"]


class WindEvent(BaseModel):
    kind: Literal["wind"]
    time: float = Field(..., allow_inf_nan=False, description="事件时刻（秒）")
    value: float = Field(
        ..., ge=0, allow_inf_nan=False, description="风速读数（>= 0）"
    )


class RainEvent(BaseModel):
    kind: Literal["rain"]
    time: float = Field(..., allow_inf_nan=False, description="事件时刻（秒）")
    value: float = Field(
        ..., ge=0, allow_inf_nan=False, description="雨量读数（>= 0）"
    )


class CloseCommandEvent(BaseModel):
    kind: Literal["close_command"]
    time: float = Field(..., allow_inf_nan=False, description="事件时刻（秒）")


class ShutterEvent(BaseModel):
    kind: Literal["shutter"]
    time: float = Field(..., allow_inf_nan=False, description="事件时刻（秒）")
    state: ShutterState = Field(
        ..., description="快门状态：open/closing/closed/opening"
    )


AuditEvent = Annotated[
    WindEvent | RainEvent | CloseCommandEvent | ShutterEvent,
    Field(discriminator="kind"),
]


class AuditSafetyInterlockRequest(BaseModel):
    events: list[AuditEvent] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="事件流（按 kind 判别：wind/rain/close_command/shutter）；"
        "时刻必须非递减，同刻事件严格按输入顺序重放",
    )
    wind_trip: float = Field(
        ..., ge=0, allow_inf_nan=False, description="风速触发阈值（>= 0）"
    )
    wind_reset: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="风速复位阈值（>= 0 且 <= wind_trip，构成迟滞）",
    )
    rain_trip: float = Field(
        ..., ge=0, allow_inf_nan=False, description="雨量触发阈值（>= 0）"
    )
    rain_reset: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="雨量复位阈值（>= 0 且 <= rain_trip，构成迟滞）",
    )
    trip_duration: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="越限需连续保持的时长（秒，>= 0；0 表示当刻触发）",
    )
    reset_duration: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="复位条件需连续保持的时长（秒，>= 0）",
    )
    stale_after: float = Field(
        ..., gt=0, allow_inf_nan=False,
        description="距最近一次天气更新严格超过该时长即判中断（秒，> 0）",
    )
    command_deadline: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="进入风险后发出关顶命令的时限（秒，>= 0）",
    )
    close_deadline: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="进入风险后确认快门 closed 的时限（秒，>= 0）",
    )

    @model_validator(mode="after")
    def _validate(self) -> "AuditSafetyInterlockRequest":
        times = [e.time for e in self.events]
        for a, b in zip(times, times[1:]):
            if b < a:
                raise ValueError(
                    "events 的 time 必须非递减（同刻事件按输入顺序处理）"
                )
        if self.wind_reset > self.wind_trip:
            raise ValueError("wind_reset 必须小于等于 wind_trip（复位阈值构成迟滞）")
        if self.rain_reset > self.rain_trip:
            raise ValueError("rain_reset 必须小于等于 rain_trip（复位阈值构成迟滞）")
        return self


class InterlockStatusSegment(BaseModel):
    start: float
    end: float | None = Field(
        ..., description="段终点（半开区间），None 表示持续到时间线末尾（开口段）"
    )
    state: WeatherState
    cause_event_indices: list[int] = Field(
        ..., description="触发本状态段的原始事件索引（0 基，输入数组下标）"
    )
    projected: bool = Field(
        ..., description="是否为末个事件之后投影得到的段（无新事件可履行其义务）"
    )


class InterlockShutterSegment(BaseModel):
    start: float
    end: float | None = Field(
        ..., description="段终点（半开区间），None 表示持续到时间线末尾；"
        "同刻多次转换会产生 start == end 的零长段"
    )
    state: ShutterState
    event_index: int = Field(..., description="产生本段的快门事件原始索引（0 基）")


class InterlockCommandRecord(BaseModel):
    time: float
    event_index: int = Field(..., description="关顶命令事件的原始索引（0 基）")


class InterlockRiskInterval(BaseModel):
    start: float
    end: float | None = Field(
        ..., description="区间终点，None 表示到时间线末尾仍未回到 safe"
    )
    states: list[Literal["unsafe", "unknown"]] = Field(
        ..., description="区间内依次出现的非安全状态（unsafe/unknown 无缝衔接合并）"
    )
    entry_event_indices: list[int] = Field(
        ..., description="进入本区间的触发事件原始索引（0 基）"
    )
    projected: bool = Field(..., description="进入时刻是否为投影产生")
    command_due: float = Field(
        ..., description="关顶命令时限时刻 = start + command_deadline"
    )
    close_due: float = Field(
        ..., description="确认 closed 时限时刻 = start + close_deadline"
    )
    command_time: float | None = Field(
        ..., description="区间内首个关顶命令时刻；未发出为 None"
    )
    command_event_index: int | None
    closed_time: float | None = Field(
        ..., description="确认 closed 的时刻；进入时已 closed 视为进入时刻确认"
    )
    closed_event_index: int | None
    already_closed_at_entry: bool = Field(
        ..., description="进入风险时快门是否已处于 closed"
    )
    command_overdue: float = Field(
        ..., description="关顶命令超期量（秒）；未履行时截算到区间末/horizon"
    )
    close_overdue: float = Field(
        ..., description="closed 确认超期量（秒）；未履行时截算到区间末/horizon"
    )


class AuditSafetyInterlockResponse(BaseModel):
    status_timeline: list[InterlockStatusSegment] = Field(
        ..., description="天气状态时间线（半开区间段，按时间顺序）"
    )
    risk_intervals: list[InterlockRiskInterval] = Field(
        ..., description="极大连续非安全区间及其关闭义务审计"
    )
    close_commands: list[InterlockCommandRecord] = Field(
        ..., description="全部关顶命令（按重放顺序）"
    )
    shutter_timeline: list[InterlockShutterSegment] = Field(
        ..., description="快门状态时间线（保留同刻零长段以反映输入顺序）"
    )
    horizon: float = Field(
        ..., description="末个事件时刻：开口义务的超期量截算到此"
    )


# ---------------------------------------------------------------------------
# POST /api/v1/project-dome-aperture
# ---------------------------------------------------------------------------


class EnuPoint(BaseModel):
    east: float = Field(..., allow_inf_nan=False, description="东向坐标 E")
    north: float = Field(..., allow_inf_nan=False, description="北向坐标 N")
    up: float = Field(..., allow_inf_nan=False, description="天顶向坐标 U")


class ApertureSample(BaseModel):
    index: int = Field(
        ..., ge=0, description="样本在 samples 数组中的 0 基下标（须与位置一致）"
    )
    azimuth: float = Field(
        ..., allow_inf_nan=False,
        description="标称视轴方位角（度，北为 0°、向东递增，可取任意实数）",
    )
    elevation: float = Field(
        ..., allow_inf_nan=False,
        description="标称视轴高度角（度，自地平面起算，须在 [-90, 90]）",
    )
    azimuth_error: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="方位角对称误差半宽（度，>= 0），误差矩形四角为 az ± 该值",
    )
    elevation_error: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="高度角对称误差半宽（度，>= 0），误差矩形四角为 el ± 该值",
    )


class ProjectDomeApertureRequest(BaseModel):
    dome_radius: float = Field(
        ..., gt=0, allow_inf_nan=False, description="球形穹顶半径（> 0，长度单位）"
    )
    origin: EnuPoint = Field(
        ..., description="视轴起点（望远镜转轴）的 ENU 坐标；必须严格位于球内"
    )
    samples: list[ApertureSample] = Field(
        ..., min_length=1, max_length=5000,
        description="待投影样本（长度 1..5000）；index 须为其 0 基数组下标",
    )

    @model_validator(mode="after")
    def _validate(self) -> "ProjectDomeApertureRequest":
        for i, s in enumerate(self.samples):
            if s.index != i:
                raise ValueError(
                    f"samples[{i}].index 必须等于其数组下标 {i}，实际为 {s.index}"
                )
        return self


class ApertureIntersection(BaseModel):
    distance: float = Field(
        ..., description="自视轴起点沿前向射线到球面交点的距离（长度单位）"
    )
    point: EnuPoint = Field(..., description="交点 ENU 坐标")
    azimuth: float = Field(
        ..., description="交点相对球心的球面方位角（度，[0, 360)，北为 0°）"
    )
    elevation: float = Field(
        ..., description="交点相对球心的球面高度角（度，[-90, 90]）"
    )


class ApertureCorner(BaseModel):
    daz: float = Field(..., description="四角方位角取值 az + daz（度）")
    dele: float = Field(..., description="四角高度角取值 el + dele（度）")
    intersection: ApertureIntersection


class ApertureEnvelope(BaseModel):
    azimuth_center: float = Field(
        ..., description="最短覆盖圆弧中心方位角（度，归一化到 [0, 360)）"
    )
    azimuth_width: float = Field(
        ..., description="最短覆盖圆弧角宽（度，[0, 360]）"
    )
    azimuth_crosses_zero: bool = Field(
        ..., description="覆盖圆弧是否跨越 0°/360° 边界"
    )
    elevation_low: float = Field(..., description="五个交点球面高度下界（度）")
    elevation_high: float = Field(..., description="五个交点球面高度上界（度）")


class ApertureSampleResult(BaseModel):
    index: int = Field(..., description="样本 0 基下标（回显）")
    nominal: ApertureIntersection = Field(
        ..., description="标称视轴前向射线与球面的最近正根交点"
    )
    corners: list[ApertureCorner] = Field(
        ...,
        description="误差矩形四角交点，固定顺序："
        "(-daz,-dele)、(+daz,-dele)、(-daz,+dele)、(+daz,+dele)",
    )
    envelope: ApertureEnvelope = Field(
        ..., description="标称 + 四角共五个交点的开口包络"
    )


class ProjectDomeApertureResponse(BaseModel):
    dome_radius: float
    origin: EnuPoint
    results: list[ApertureSampleResult]


# ---------------------------------------------------------------------------
# POST /api/v1/plan-dome-balance
# ---------------------------------------------------------------------------


class InstalledPart(BaseModel):
    mass: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="已装部件质量（>= 0；零质量部件不贡献质量矩）",
    )
    radius: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="部件质心到旋转轴的半径（>= 0）",
    )
    azimuth: float = Field(
        ..., allow_inf_nan=False,
        description="部件相对旋转轴的方位角（度，北为 0°、向东递增，可取任意实数）",
    )


class BalanceSlot(BaseModel):
    id: str = Field(
        ..., min_length=1, description="槽位 ID（非空字符串，请求内唯一）"
    )
    radius: float = Field(
        ..., ge=0, allow_inf_nan=False, description="槽位到旋转轴的半径（>= 0）"
    )
    azimuth: float = Field(
        ..., allow_inf_nan=False,
        description="槽位方位角（度，北为 0°、向东递增，可取任意实数）",
    )
    mass_limit: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="该槽位允许的最大配重质量（>= 0；至少需容纳一个 mass_step 档位）",
    )


class PlanDomeBalanceRequest(BaseModel):
    installed_parts: list[InstalledPart] = Field(
        ..., min_length=1, description="已装部件清单（至少 1 个）"
    )
    slots: list[BalanceSlot] = Field(
        ..., min_length=1, max_length=BALANCE_MAX_SLOTS,
        description="候选配重槽位（1..8 个，ID 唯一）",
    )
    mass_step: float = Field(
        ..., gt=0, allow_inf_nan=False,
        description="统一配重质量档位步长（> 0）；每槽可选质量为 "
        "0, mass_step, 2·mass_step, … 且不超过该槽 mass_limit",
    )
    mass_tolerance_ratio: float = Field(
        ..., ge=0, lt=1, allow_inf_nan=False,
        description="质量相对偏差（>= 0 且 < 1）；实际质量可取 "
        "标称·(1 ± 该值)",
    )
    azimuth_tolerance: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="方位偏差半宽（度，>= 0）；实际方位可取标称 ± 该值",
    )
    residual_moment_limit: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="残余质量矩模长验收上限（>= 0）；最坏情景模长不超过它即达标",
    )

    @model_validator(mode="after")
    def _validate(self) -> "PlanDomeBalanceRequest":
        ids = [s.id for s in self.slots]
        if len(set(ids)) != len(ids):
            raise ValueError("slots 的 id 必须唯一")
        for s in self.slots:
            # 与 planner.balance.slot_levels 相同的取档规则
            n_levels = int(math.floor(s.mass_limit / self.mass_step + 1e-9))
            if n_levels < 1:
                raise ValueError(
                    f"槽位 {s.id} 的 mass_limit={s.mass_limit} 小于 "
                    f"mass_step={self.mass_step}，至少需容纳一个非零档位"
                )
            if n_levels > BALANCE_MAX_LEVELS_PER_SLOT:
                raise ValueError(
                    f"槽位 {s.id} 的非零质量档位数 {n_levels} 超过上限 "
                    f"{BALANCE_MAX_LEVELS_PER_SLOT}"
                )
        return self


class MomentVector2(BaseModel):
    east: float = Field(..., description="质量矩东向分量 x（质量·长度）")
    north: float = Field(..., description="质量矩北向分量 y（质量·长度）")


class BalanceWeightItem(BaseModel):
    slot_id: str
    radius: float = Field(..., description="槽位半径（回显）")
    nominal_mass: float = Field(..., description="该槽位选定的标称配重质量")
    nominal_azimuth: float = Field(
        ..., description="槽位标称方位角（度，按请求回显，未做 0..360 归一化）"
    )
    worst_mass: float = Field(
        ..., description="最坏偏差情景下该配重的实际质量"
    )
    worst_azimuth: float = Field(
        ..., description="最坏偏差情景下该配重的实际方位角（度，"
        "= 标称方位 + azimuth_sign·azimuth_tolerance，未归一化）"
    )


class WorstDeviationItem(BaseModel):
    slot_id: str
    mass: float = Field(..., description="该配重在最坏组合中的实际质量")
    azimuth: float = Field(
        ..., description="该配重在最坏组合中的实际方位角（度，未归一化）"
    )
    mass_sign: int = Field(
        ..., description="质量偏差符号：-1 负偏差、0 标称、+1 正偏差"
    )
    azimuth_sign: int = Field(
        ..., description="方位偏差符号：-1 负偏差、0 标称、+1 正偏差"
    )


class PlanDomeBalanceResponse(BaseModel):
    original_moment: MomentVector2 = Field(
        ..., description="全部已装部件合成的原始质量矩向量"
    )
    original_magnitude: float = Field(..., description="原始质量矩模长")
    weights: list[BalanceWeightItem] = Field(
        ..., description="配重清单（按槽位 ID 排序；零配重方案为空列表）"
    )
    total_balance_mass: float = Field(
        ..., description="配重总质量 Σ nominal_mass"
    )
    nominal_residual: MomentVector2 = Field(
        ..., description="标称配重下的残余质量矩向量"
    )
    nominal_magnitude: float = Field(
        ..., description="标称残余质量矩模长"
    )
    worst_residual: MomentVector2 = Field(
        ..., description="最坏偏差情景下的残余质量矩向量"
    )
    worst_magnitude: float = Field(
        ..., description="最坏残余质量矩模长（所有偏差组合中的最大模长）"
    )
    worst_deviation_combo: list[WorstDeviationItem] = Field(
        ..., description="达到最坏模长的偏差组合（与 weights 同序；"
        "多个组合并列时取偏差符号字典序最小者）；零配重方案为空列表"
    )
    residual_moment_limit: float = Field(..., description="验收限值（回显）")
    compliant: bool = Field(
        ..., description="最坏残余质量矩模长是否不超过验收限值"
    )
    details: dict = Field(
        ..., description="枚举诊断：evaluated_plans（穷举方案数）、"
        "max_scenarios_per_plan（单方案最多偏差情景数）、"
        "mass_vector（按槽位 ID 排列的质量向量，未使用槽位记 0）、"
        "slot_ids（质量向量各分量对应的槽位 ID）"
    )


# ---------------------------------------------------------------------------
# POST /api/v1/plan-weight-sequence
# ---------------------------------------------------------------------------


class OperableWeight(BaseModel):
    id: str = Field(
        ..., min_length=1, description="可操作配重 ID（非空字符串，请求内唯一）"
    )
    mass: float = Field(
        ..., ge=0, allow_inf_nan=False, description="配重质量（>= 0）"
    )
    radius: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="配重质心到旋转轴的半径（>= 0）",
    )
    azimuth: float = Field(
        ..., allow_inf_nan=False,
        description="配重相对旋转轴的方位角（度，北为 0°、向东递增，"
        "可取任意实数，跨 0° 无需特判）",
    )


class PlanWeightSequenceRequest(BaseModel):
    fixed_parts: list[InstalledPart] = Field(
        default_factory=list,
        description="固定部件清单（不可拆装，质量与质量矩始终在底座上）；"
        "可为空",
    )
    weights: list[OperableWeight] = Field(
        ..., max_length=RESEQUENCE_MAX_WEIGHTS,
        description="可操作配重（**最多 12 件**，ID 非空且唯一）："
        "每件具有质量、半径与方位",
    )
    current_ids: list[str] = Field(
        ..., description="当前在位配重 ID 列表（均须为已知 ID，重复自动去重）"
    )
    target_ids: list[str] = Field(
        ..., description="目标在位配重 ID 列表（均须为已知 ID，重复自动去重）"
    )
    installed_mass_limit: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="在位总质量上限（固定部件 + 在位配重，>= 0）；"
        "每步结束（含起止状态）均不得越限",
    )
    residual_moment_limit: float = Field(
        ..., ge=0, allow_inf_nan=False,
        description="残余质量矩模长上限（>= 0）；每步结束（含起止状态）"
        "均不得越限",
    )

    @model_validator(mode="after")
    def _validate(self) -> "PlanWeightSequenceRequest":
        ids = [w.id for w in self.weights]
        if len(set(ids)) != len(ids):
            raise ValueError("weights 的 id 必须唯一")
        known = set(ids)
        unknown_current = [i for i in self.current_ids if i not in known]
        if unknown_current:
            raise ValueError(
                f"current_ids 包含未知配重 ID：{unknown_current}"
            )
        unknown_target = [i for i in self.target_ids if i not in known]
        if unknown_target:
            raise ValueError(
                f"target_ids 包含未知配重 ID：{unknown_target}"
            )
        return self


class WeightSequenceState(BaseModel):
    installed_ids: list[str] = Field(
        ..., description="在位配重 ID 列表（升序）"
    )
    total_mass: float = Field(..., description="在位总质量（固定部件 + 配重）")
    residual: MomentVector2 = Field(..., description="残余质量矩向量")
    residual_magnitude: float = Field(..., description="残余质量矩模长")


class WeightSequenceStep(BaseModel):
    index: int = Field(..., description="步号（1 基）")
    action: Literal["install", "remove"] = Field(
        ..., description="动作类型：install（装上）/ remove（拆下）"
    )
    weight_id: str = Field(..., description="本步操作的配重 ID")
    installed_ids: list[str] = Field(
        ..., description="本步结束后的在位配重 ID（升序）"
    )
    total_mass: float = Field(..., description="本步结束后的在位总质量")
    residual: MomentVector2 = Field(
        ..., description="本步结束后的残余质量矩向量"
    )
    residual_magnitude: float = Field(
        ..., description="本步结束后的残余质量矩模长"
    )
    temporary: bool = Field(
        ..., description="本步操作的配重是否为临时配重"
        "（起点不在位、目标集合外，中途借来且最终拆除）"
    )


class BlockedWeightAction(BaseModel):
    action: Literal["install", "remove"] = Field(
        ..., description="被阻断动作类型：install / remove"
    )
    weight_id: str = Field(..., description="被阻断动作操作的配重 ID")
    from_depth: int = Field(
        ..., description="来源状态距起点的动作步数（0 即起点）"
    )
    from_installed_ids: list[str] = Field(
        ..., description="动作执行前来源状态的在位配重 ID（升序）"
    )
    resulting_installed_ids: list[str] = Field(
        ..., description="动作若执行后在位配重 ID（升序；该状态越限）"
    )
    resulting_total_mass: float = Field(
        ..., description="动作若执行后的在位总质量（已越限）"
    )
    resulting_residual: MomentVector2 = Field(
        ..., description="动作若执行后的残余质量矩向量（已越限）"
    )
    resulting_residual_magnitude: float = Field(
        ..., description="动作若执行后的残余质量矩模长（已越限）"
    )
    moment_excess: float = Field(
        ..., description="质量矩超限量（= max(0, 模长 − 限值)，边界容差 1e-9）"
    )
    mass_excess: float = Field(
        ..., description="总质量超限量（= max(0, 总质量 − 限值)，边界容差 1e-9）"
    )


class PlanWeightSequenceResponse(BaseModel):
    feasible: bool = Field(
        ..., description="是否存在满足全部安全约束的换装路径"
    )
    fixed_total_mass: float = Field(..., description="固定部件总质量")
    fixed_moment: MomentVector2 = Field(
        ..., description="固定部件合成质量矩向量"
    )
    fixed_moment_magnitude: float = Field(
        ..., description="固定部件合成质量矩模长"
    )
    installed_mass_limit: float = Field(..., description="在位总质量上限（回显）")
    residual_moment_limit: float = Field(
        ..., description="残余质量矩模长上限（回显）"
    )
    initial_state: WeightSequenceState = Field(..., description="起点状态")
    target_state: WeightSequenceState = Field(..., description="终点状态")
    target_ids: list[str] = Field(
        ..., description="目标在位配重 ID（升序回显）"
    )
    steps: list[WeightSequenceStep] = Field(
        default_factory=list,
        description="逐步换装动作（动作最少；无路时为空列表）",
    )
    temporary_weight_ids: list[str] = Field(
        default_factory=list,
        description="路径上使用过的临时配重 ID（升序）：起点不在位且目标外、"
        "中途借来最终拆除的配重；换装结束时它们均已拆除，终态恰为目标集合",
    )
    reachable_states: int = Field(
        ..., description="从起点可达的合法状态数（含起点）"
    )
    max_intermediate_magnitude: float = Field(
        0.0,
        description="所选路径各步结束残余模长的最大值（同长度择优键；"
        "无路时为 0）",
    )
    nearest_blocked_action: BlockedWeightAction | None = Field(
        None,
        description="无路时从可达状态出发的“最近一步”被阻断动作"
        "（依次按质量矩超限量、总质量超限量、动作字典序取最小）；"
        "有解时为 null",
    )
    details: dict = Field(
        default_factory=dict,
        description="诊断：explored_states（穷举的掩码状态总数，= 2^配重建数）",
    )
