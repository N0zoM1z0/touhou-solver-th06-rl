"""Low-cost Windows commit telemetry for protecting the physical host."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os


GIB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class SystemMemorySample:
    commit_total_bytes: int
    commit_limit_bytes: int
    physical_available_bytes: int
    controller_private_bytes: int

    @property
    def commit_headroom_bytes(self) -> int:
        return max(0, self.commit_limit_bytes - self.commit_total_bytes)


class _PerformanceInformation(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    )


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    )


def read_system_memory() -> SystemMemorySample:
    """Read system commit and this controller's private commit on Windows."""
    if os.name != "nt":
        raise OSError("Windows memory telemetry is unavailable on this host")
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi.GetPerformanceInfo.argtypes = (
        ctypes.POINTER(_PerformanceInformation),
        ctypes.c_ulong,
    )
    psapi.GetPerformanceInfo.restype = ctypes.c_int
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p

    performance = _PerformanceInformation()
    performance.cb = ctypes.sizeof(performance)
    if not psapi.GetPerformanceInfo(
        ctypes.byref(performance), ctypes.sizeof(performance)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        ctypes.sizeof(counters),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    page = int(performance.PageSize)
    return SystemMemorySample(
        commit_total_bytes=int(performance.CommitTotal) * page,
        commit_limit_bytes=int(performance.CommitLimit) * page,
        physical_available_bytes=int(performance.PhysicalAvailable) * page,
        controller_private_bytes=int(counters.PrivateUsage),
    )


def below_commit_reserve(sample: SystemMemorySample, reserve_bytes: int) -> bool:
    if reserve_bytes < 0:
        raise ValueError("commit reserve cannot be negative")
    return sample.commit_headroom_bytes < reserve_bytes
