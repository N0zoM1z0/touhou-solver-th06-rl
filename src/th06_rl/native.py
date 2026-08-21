"""ctypes facade for the compact native observed-hazard shield kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from .core.model import Action, CertifiedAction, Kinematics, movement_actions


ACTIONS = movement_actions()
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


@dataclass(frozen=True)
class Aabb:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class LaserRect:
    origin_x: float
    origin_y: float
    angle: float
    center_offset: float
    size_x: float
    size_y: float


@dataclass(frozen=True)
class PackedHazards:
    aabb_frames: tuple[tuple[Aabb, ...], ...]
    laser_frames: tuple[tuple[LaserRect, ...], ...]

    def __post_init__(self) -> None:
        if not self.aabb_frames:
            raise ValueError("hazard forecast must contain at least one frame")
        if len(self.aabb_frames) != len(self.laser_frames):
            raise ValueError("AABB and laser forecasts must have equal horizons")

    @property
    def horizon(self) -> int:
        return len(self.aabb_frames)


@dataclass(frozen=True)
class NativeCertifiedAction:
    action: Action
    min_clearance: float
    final_x: float
    final_y: float

    @property
    def core(self) -> CertifiedAction:
        return CertifiedAction(self.action, self.min_clearance)


class _Aabb(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_float),
        ("top", ctypes.c_float),
        ("right", ctypes.c_float),
        ("bottom", ctypes.c_float),
    ]


class _LaserRect(ctypes.Structure):
    _fields_ = [
        ("origin_x", ctypes.c_float),
        ("origin_y", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("center_offset", ctypes.c_float),
        ("size_x", ctypes.c_float),
        ("size_y", ctypes.c_float),
    ]


@dataclass(frozen=True)
class PreparedHazards:
    """One immutable ctypes packing with cheap horizon-prefix views."""

    horizon: int
    aabb_offsets: object
    aabbs: object
    laser_offsets: object
    lasers: object

    def prefix(self, horizon: int) -> PreparedHazards:
        if not 1 <= horizon <= self.horizon:
            raise ValueError(
                f"prepared hazard prefix {horizon} is outside 1..{self.horizon}"
            )
        return PreparedHazards(
            horizon,
            self.aabb_offsets,
            self.aabbs,
            self.laser_offsets,
            self.lasers,
        )

    @property
    def native_args(self) -> tuple[object, object, object, object]:
        return (
            self.aabb_offsets,
            self.aabbs,
            self.laser_offsets,
            self.lasers,
        )


def _action_mask(actions: Iterable[Action]) -> int:
    mask = 0
    for action in actions:
        try:
            index = ACTION_INDEX[action]
        except KeyError as error:
            raise ValueError(f"unknown native action {action!r}") from error
        mask |= 1 << index
    return mask


def _default_library_path() -> Path:
    override = os.environ.get("TH06_RL_NATIVE_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    names = (
        "th06_rl_native.dll",
        "libth06_rl_native.so",
        "libth06_rl_native.dylib",
    )
    candidates = tuple(
        root / directory / name
        for directory in ("build/native/Release", "build/native", "build")
        for name in names
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class NativeKernel:
    def __init__(self, library_path: Path | None = None) -> None:
        path = (library_path or _default_library_path()).resolve()
        if not path.is_file():
            raise RuntimeError(f"native TH06-RL kernel is missing: {path}")
        self.path = path
        self.library = ctypes.CDLL(str(path))
        self._configure()

    def _configure(self) -> None:
        floats8 = [ctypes.c_float] * 8
        self.certify_function = self.library.th06_rl_certify_actions_v1
        self.certify_function.argtypes = [
            *floats8,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_Aabb),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_LaserRect),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.certify_function.restype = ctypes.c_int

    @staticmethod
    def _pack(
        hazards: PackedHazards,
    ) -> tuple[object, object, object, object]:
        aabbs = tuple(item for frame in hazards.aabb_frames for item in frame)
        lasers = tuple(item for frame in hazards.laser_frames for item in frame)
        aabb_array = (_Aabb * max(1, len(aabbs)))(
            *(_Aabb(*item.__dict__.values()) for item in aabbs)
        )
        laser_array = (_LaserRect * max(1, len(lasers)))(
            *(_LaserRect(*item.__dict__.values()) for item in lasers)
        )
        aabb_offsets = [0]
        laser_offsets = [0]
        for aabb_frame, laser_frame in zip(
            hazards.aabb_frames,
            hazards.laser_frames,
            strict=True,
        ):
            aabb_offsets.append(aabb_offsets[-1] + len(aabb_frame))
            laser_offsets.append(laser_offsets[-1] + len(laser_frame))
        offset_type = ctypes.c_uint32 * (hazards.horizon + 1)
        return (
            offset_type(*aabb_offsets),
            aabb_array,
            offset_type(*laser_offsets),
            laser_array,
        )

    @classmethod
    def prepare_hazards(cls, hazards: PackedHazards) -> PreparedHazards:
        return PreparedHazards(
            hazards.horizon,
            *cls._pack(hazards),
        )

    @staticmethod
    def _physical_args(
        x: float,
        y: float,
        half_width: float,
        half_height: float,
        kinematics: Kinematics,
    ) -> tuple[float, ...]:
        return (
            x,
            y,
            half_width,
            half_height,
            kinematics.normal_speed,
            kinematics.focus_speed,
            kinematics.normal_diagonal_speed,
            kinematics.focus_diagonal_speed,
        )

    def certify_actions(
        self,
        *,
        x: float,
        y: float,
        half_width: float,
        half_height: float,
        kinematics: Kinematics,
        current_action: Action,
        hazards: PackedHazards | PreparedHazards,
        candidates: tuple[Action, ...] = ACTIONS,
        delivery_delays: tuple[int, ...] = (0, 1, 2, 3),
        collision_margin: float = 0.35,
    ) -> tuple[NativeCertifiedAction, ...]:
        if not delivery_delays:
            raise ValueError("delivery delays cannot be empty")
        prepared = (
            hazards
            if isinstance(hazards, PreparedHazards)
            else self.prepare_hazards(hazards)
        )
        delay_array = (ctypes.c_int32 * len(delivery_delays))(*delivery_delays)
        safe_mask = ctypes.c_uint32()
        minimum = (ctypes.c_float * len(ACTIONS))()
        final_xy = (ctypes.c_float * (len(ACTIONS) * 2))()
        common_args = (
            *self._physical_args(x, y, half_width, half_height, kinematics),
            ACTION_INDEX[current_action],
            prepared.horizon,
            delay_array,
            len(delivery_delays),
            _action_mask(candidates),
            *prepared.native_args,
        )
        status = self.certify_function(
            *common_args,
            collision_margin,
            ctypes.byref(safe_mask),
            minimum,
            final_xy,
        )
        if status != 0:
            raise RuntimeError(f"native shield kernel rejected input: {status}")
        return tuple(
            NativeCertifiedAction(
                action,
                float(minimum[index]),
                float(final_xy[index * 2]),
                float(final_xy[index * 2 + 1]),
            )
            for index, action in enumerate(ACTIONS)
            if safe_mask.value & (1 << index)
        )
