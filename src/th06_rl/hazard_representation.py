"""Game-neutral bounded hazard sets and factual observation history."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import math
from pathlib import Path


HAZARD_PRIMITIVE_SCHEMA = "game-neutral-observed-hazard-primitives-v1"
HAZARD_PRIMITIVE_FEATURE_NAMES = (
    "relative_x_unit",
    "relative_y_unit",
    "velocity_x_unit",
    "velocity_y_unit",
    "extent_x_unit",
    "extent_y_unit",
    "distance_unit",
    "closing_speed_unit",
    "age_log",
    "remaining_life_log",
    "remaining_life_known",
    "kind_point",
    "kind_segment",
    "kind_body",
)
MAX_HAZARD_PRIMITIVES = 256
HAZARD_CODEBOOK_SCHEMA = "game-neutral-hazard-codebook-v1"
HAZARD_CODEBOOK_PROTOTYPES = 24
HISTORY_OBSERVATIONS = 4
_HISTORY_FIELDS = (
    "present",
    "age_frames_log",
    "player_x_unit",
    "player_y_unit",
    "bullet_count_log",
    "laser_count_log",
    "action_direction_x",
    "action_direction_y",
    "action_focused",
)
HISTORY_FEATURE_NAMES = tuple(
    f"lag{lag}:{name}"
    for lag in range(HISTORY_OBSERVATIONS)
    for name in _HISTORY_FIELDS
)


def hazard_codebook_feature_names(
    prototype_count: int = HAZARD_CODEBOOK_PROTOTYPES,
) -> tuple[str, ...]:
    if prototype_count <= 0:
        raise ValueError("hazard codebook needs at least one prototype")
    return (
        *(f"hazard:prototype_fraction_{index}" for index in range(prototype_count)),
        *(
            f"hazard:prototype_min_distance_{index}"
            for index in range(prototype_count)
        ),
        *(f"hazard:mean_{name}" for name in HAZARD_PRIMITIVE_FEATURE_NAMES),
        *(f"hazard:max_abs_{name}" for name in HAZARD_PRIMITIVE_FEATURE_NAMES),
        "hazard:count_log",
        "hazard:empty",
    )


def encode_hazard_set(
    primitives: tuple[tuple[float, ...], ...],
    artifact: dict[str, object],
) -> tuple[float, ...]:
    """Portable reference for the retained native codebook encoder."""
    width = len(HAZARD_PRIMITIVE_FEATURE_NAMES)
    prototype_count = int(artifact.get("prototype_count", -1))
    mean = tuple(float(value) for value in artifact.get("mean", ()))
    scale = tuple(float(value) for value in artifact.get("scale", ()))
    prototypes = tuple(
        tuple(float(value) for value in row)
        for row in artifact.get("prototypes", ())
    )
    if (
        artifact.get("schema") != HAZARD_CODEBOOK_SCHEMA
        or tuple(artifact.get("primitive_feature_names", ()))
        != HAZARD_PRIMITIVE_FEATURE_NAMES
        or prototype_count <= 0
        or len(mean) != width
        or len(scale) != width
        or len(prototypes) != prototype_count
        or any(len(row) != width for row in prototypes)
        or any(not math.isfinite(value) for value in mean)
        or any(not math.isfinite(value) or value <= 0.0 for value in scale)
        or any(not math.isfinite(value) for row in prototypes for value in row)
        or len(primitives) > MAX_HAZARD_PRIMITIVES
        or any(len(row) != width for row in primitives)
    ):
        raise ValueError("hazard codebook contract mismatch")
    normalized = tuple(
        tuple(
            (float(value) - center) / spread
            for value, center, spread in zip(row, mean, scale, strict=True)
        )
        for row in primitives
    )
    distances = tuple(
        tuple(
            sum(
                (value - prototype_value) ** 2
                for value, prototype_value in zip(row, prototype, strict=True)
            ) / width
            for prototype in prototypes
        )
        for row in normalized
    )
    if normalized:
        assignments = tuple(
            min(range(prototype_count), key=row.__getitem__) for row in distances
        )
        fractions = tuple(
            assignments.count(index) / len(assignments)
            for index in range(prototype_count)
        )
        minimum = tuple(
            min(row[index] for row in distances)
            for index in range(prototype_count)
        )
        average = tuple(
            sum(row[index] for row in normalized) / len(normalized)
            for index in range(width)
        )
        max_abs = tuple(
            max(abs(row[index]) for row in normalized) for index in range(width)
        )
    else:
        fractions = (0.0,) * prototype_count
        minimum = (0.0,) * prototype_count
        average = (0.0,) * width
        max_abs = (0.0,) * width
    encoded = (
        *fractions,
        *minimum,
        *average,
        *max_abs,
        math.log1p(len(primitives)),
        float(not primitives),
    )
    if len(encoded) != len(hazard_codebook_feature_names(prototype_count)):
        raise RuntimeError("hazard codebook encoding width mismatch")
    return tuple(encoded)


@dataclass(frozen=True)
class HistoryObservation:
    frame: int
    stage: int
    player_x_unit: float
    player_y_unit: float
    bullet_count_log: float
    laser_count_log: float
    action_direction_x: float
    action_direction_y: float
    action_focused: float


def _finite(value: object, default: float = 0.0) -> float:
    result = float(value)
    return result if math.isfinite(result) else default


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _action_components(action: str) -> tuple[float, float, float]:
    focused = float(not action.endswith("_fast"))
    core = action.removesuffix("_fast")
    dx = float("left" in core) * -1.0 + float("right" in core)
    dy = float("up" in core) * -1.0 + float("down" in core)
    return dx, dy, focused


def make_history_observation(
    snapshot,
    action: str,
    *,
    left: float = 8.0,
    right: float = 376.0,
    top: float = 16.0,
    bottom: float = 432.0,
) -> HistoryObservation:
    width = right - left
    height = bottom - top
    if min(width, height) <= 0.0:
        raise ValueError("playfield bounds are invalid")
    dx, dy, focused = _action_components(action)
    return HistoryObservation(
        frame=int(snapshot.frame),
        stage=int(snapshot.stage),
        player_x_unit=(_finite(snapshot.x) - left) / width,
        player_y_unit=(_finite(snapshot.y) - top) / height,
        bullet_count_log=math.log1p(max(0, int(snapshot.live_bullet_count))),
        laser_count_log=math.log1p(max(0, int(snapshot.laser_count))),
        action_direction_x=dx,
        action_direction_y=dy,
        action_focused=focused,
    )


def project_history_features(
    current: HistoryObservation,
    previous: tuple[HistoryObservation, ...],
) -> tuple[tuple[str, float], ...]:
    rows = [current]
    rows.extend(
        item for item in reversed(previous)
        if item.stage == current.stage and item.frame < current.frame
    )
    values = []
    for lag in range(HISTORY_OBSERVATIONS):
        if lag < len(rows):
            item = rows[lag]
            raw = (
                1.0,
                math.log1p(current.frame - item.frame),
                item.player_x_unit,
                item.player_y_unit,
                item.bullet_count_log,
                item.laser_count_log,
                item.action_direction_x,
                item.action_direction_y,
                item.action_focused,
            )
        else:
            raw = (0.0,) * len(_HISTORY_FIELDS)
        values.extend(raw)
    return tuple(zip(HISTORY_FEATURE_NAMES, values, strict=True))


def _primitive(
    *,
    player_x: float,
    player_y: float,
    x: float,
    y: float,
    vx: float,
    vy: float,
    extent_x: float,
    extent_y: float,
    age: float,
    remaining: float | None,
    kind: int,
    width: float,
    height: float,
    velocity_scale: float,
) -> tuple[float, ...]:
    relative_x = (_finite(x) - player_x) / width
    relative_y = (_finite(y) - player_y) / height
    velocity_x = _finite(vx) / velocity_scale
    velocity_y = _finite(vy) / velocity_scale
    distance = math.hypot(relative_x, relative_y)
    closing = -(
        relative_x * velocity_x + relative_y * velocity_y
    ) / max(distance, 1e-6)
    return (
        relative_x,
        relative_y,
        velocity_x,
        velocity_y,
        abs(_finite(extent_x)) / width,
        abs(_finite(extent_y)) / height,
        distance,
        closing,
        _signed_log1p(max(0.0, _finite(age))),
        _signed_log1p(max(0.0, _finite(remaining)))
        if remaining is not None else 0.0,
        float(remaining is not None),
        float(kind == 0),
        float(kind == 1),
        float(kind == 2),
    )


def project_hazard_primitives(
    snapshot,
    *,
    left: float = 8.0,
    right: float = 376.0,
    top: float = 16.0,
    bottom: float = 432.0,
    velocity_scale: float = 16.0,
) -> tuple[tuple[float, ...], ...]:
    """Project observed hazards without source identity or control-flow keys."""
    width = right - left
    height = bottom - top
    if min(width, height, velocity_scale) <= 0.0:
        raise ValueError("hazard normalization bounds are invalid")
    player_x, player_y = _finite(snapshot.x), _finite(snapshot.y)
    result = []
    for bullet in snapshot.bullets:
        result.append(_primitive(
            player_x=player_x,
            player_y=player_y,
            x=bullet.x,
            y=bullet.y,
            vx=bullet.vx,
            vy=bullet.vy,
            extent_x=bullet.half_width,
            extent_y=bullet.half_height,
            age=bullet.timer,
            remaining=None,
            kind=0,
            width=width,
            height=height,
            velocity_scale=velocity_scale,
        ))
    for laser in snapshot.lasers:
        angle = _finite(laser.angle)
        start = _finite(laser.start_offset)
        end = _finite(laser.end_offset)
        midpoint = (start + end) / 2.0
        length = abs(end - start)
        cosine, sine = math.cos(angle), math.sin(angle)
        timer = max(0.0, _finite(laser.timer))
        duration = max(0.0, _finite(laser.duration))
        result.append(_primitive(
            player_x=player_x,
            player_y=player_y,
            x=_finite(laser.x) + cosine * midpoint,
            y=_finite(laser.y) + sine * midpoint,
            vx=cosine * _finite(laser.speed),
            vy=sine * _finite(laser.speed),
            extent_x=abs(cosine) * length / 2.0 + abs(sine) * _finite(laser.width) / 2.0,
            extent_y=abs(sine) * length / 2.0 + abs(cosine) * _finite(laser.width) / 2.0,
            age=timer,
            remaining=max(0.0, duration - timer),
            kind=1,
            width=width,
            height=height,
            velocity_scale=velocity_scale,
        ))
    for body in snapshot.enemies:
        result.append(_primitive(
            player_x=player_x,
            player_y=player_y,
            x=body.x,
            y=body.y,
            vx=body.velocity_x,
            vy=body.velocity_y,
            extent_x=body.half_width,
            extent_y=body.half_height,
            age=body.move_timer,
            remaining=None,
            kind=2,
            width=width,
            height=height,
            velocity_scale=velocity_scale,
        ))
    if any(
        len(row) != len(HAZARD_PRIMITIVE_FEATURE_NAMES)
        or not all(math.isfinite(value) for value in row)
        for row in result
    ):
        raise ValueError("hazard primitive projection produced invalid values")
    result.sort(key=lambda row: (row[6], row[11:14], row))
    return tuple(result[:MAX_HAZARD_PRIMITIVES])


class NativeHazardCodebookEncoder:
    """Bounded native facade for an immutable learned primitive codebook."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        artifact: dict[str, object],
        output_count: int,
    ) -> None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("native hazard encoder SHA-256 mismatch")
        mean = tuple(float(value) for value in artifact.get("mean", ()))
        scale = tuple(float(value) for value in artifact.get("scale", ()))
        prototypes = tuple(
            tuple(float(value) for value in row)
            for row in artifact.get("prototypes", ())
        )
        width = len(HAZARD_PRIMITIVE_FEATURE_NAMES)
        if (
            artifact.get("schema") != HAZARD_CODEBOOK_SCHEMA
            or tuple(artifact.get("primitive_feature_names", ()))
            != HAZARD_PRIMITIVE_FEATURE_NAMES
            or len(mean) != width
            or len(scale) != width
            or not prototypes
            or any(len(row) != width for row in prototypes)
            or output_count != 2 * len(prototypes) + 2 * width + 2
        ):
            raise ValueError("native hazard encoder artifact shape mismatch")
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_encode_hazard_codebook_v1
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ]
        function.restype = ctypes.c_int
        self.library = library
        self.function = function
        self.mean = (ctypes.c_float * width)(*mean)
        self.scale = (ctypes.c_float * width)(*scale)
        flat = tuple(value for row in prototypes for value in row)
        self.prototypes = (ctypes.c_float * len(flat))(*flat)
        self.feature_count = width
        self.prototype_count = len(prototypes)
        self.output_count = output_count

    def encode(
        self, primitives: tuple[tuple[float, ...], ...]
    ) -> tuple[float, ...]:
        if len(primitives) > MAX_HAZARD_PRIMITIVES or any(
            len(row) != self.feature_count for row in primitives
        ):
            raise ValueError("native hazard encoder input shape mismatch")
        flat_values = tuple(value for row in primitives for value in row)
        flat = (ctypes.c_float * max(1, len(flat_values)))(
            *(flat_values or (0.0,))
        )
        output = (ctypes.c_float * self.output_count)()
        status = self.function(
            flat,
            len(primitives),
            self.feature_count,
            self.mean,
            self.scale,
            self.prototypes,
            self.prototype_count,
            output,
            self.output_count,
        )
        if status != 0:
            raise RuntimeError(f"native hazard encoder failed with status {status}")
        return tuple(float(value) for value in output)
