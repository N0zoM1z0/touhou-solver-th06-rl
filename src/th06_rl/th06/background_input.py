"""Background TH06 input at the single authoritative Supervisor call site.

The supported executable calls ``Controller::GetInput`` at 0x00423361 and
immediately stores AX into ``g_CurFrameInput``.  We replace that complete
five-byte CALL with another five-byte CALL to a tiny controller-owned stub.
Unlike the retired epilogue hook, this never creates an entry point in the
middle of an instruction used by GetInput's focus-loss paths.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from ctypes import wintypes
import os
import struct
import time

from .donor import enable_donor_imports

enable_donor_imports()
from th06.model import (  # noqa: E402
    BUTTON_BOMB,
    BUTTON_DOWN,
    BUTTON_FOCUS,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_SHOOT,
    BUTTON_SKIP,
    BUTTON_UP,
)


MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_EXECUTE_READWRITE = 0x40
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259

# Supervisor::OnUpdate:
#   00423361  E8 BA A4 FF FF  call Controller::GetInput
#   00423366  66 A3 04 D9 69 00  mov g_CurFrameInput, ax
HOOK_ADDRESS = 0x00423361
HOOK_ORIGINAL = bytes.fromhex("E8 BA A4 FF FF")
HOOK_SIZE = len(HOOK_ORIGINAL)
CAVE_SIZE = 0x100
CONTROL_OFFSET = 0x80
CONTROL_MAGIC = b"TH06RLI2"
CONTROL_STRUCT = struct.Struct("<II8s")

# Controller.hpp: Bomb is 0x02. It is absent here and rejected on every write.
BUTTON_MENU = 0x0008
KEY_MASKS = {
    "shoot": BUTTON_SHOOT,
    "focus": BUTTON_FOCUS,
    "skip": BUTTON_SKIP,
    "menu": BUTTON_MENU,
    "up": BUTTON_UP,
    "down": BUTTON_DOWN,
    "left": BUTTON_LEFT,
    "right": BUTTON_RIGHT,
}
ALLOWED_INPUT_MASK = sum(KEY_MASKS.values())

if ALLOWED_INPUT_MASK & BUTTON_BOMB:
    raise AssertionError("background input mask must never contain Bomb")


def rel32(source: int, target: int) -> bytes:
    displacement = target - (source + 5)
    if not -(2**31) <= displacement < 2**31:
        raise RuntimeError(f"relative branch out of range: {source:#x} -> {target:#x}")
    return struct.pack("<i", displacement)


def call_target(source: int, instruction: bytes) -> int:
    if len(instruction) != 5 or instruction[0] != 0xE8:
        raise ValueError("expected a five-byte near call")
    return source + 5 + struct.unpack("<i", instruction[1:])[0]


def keys_to_input_mask(names: set[str]) -> int:
    unknown = names - KEY_MASKS.keys()
    if unknown:
        raise ValueError(f"unknown TH06 keys: {sorted(unknown)}")
    mask = 0
    for name in names:
        mask |= KEY_MASKS[name]
    if mask & BUTTON_BOMB:
        raise AssertionError("Bomb cannot be encoded by the input bridge")
    return mask


def build_stub(cave: int) -> bytes:
    control = cave + CONTROL_OFFSET
    # mov eax,[control.mask]; and eax,ALLOWED_INPUT_MASK; ret
    code = (
        b"\xA1"
        + struct.pack("<I", control)
        + b"\x25"
        + struct.pack("<I", ALLOWED_INPUT_MASK)
        + b"\xC3"
    )
    if len(code) >= CONTROL_OFFSET:
        raise AssertionError("input stub overlaps its control block")
    return code


def build_hook(cave: int) -> bytes:
    return b"\xE8" + rel32(HOOK_ADDRESS, cave)


class BackgroundInputBridge:
    """Reversible, exact-PID background input with an in-process Bomb mask."""

    def __init__(self, process) -> None:
        self.process = process
        self.kernel32 = process.kernel32
        self.ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self.cave: int | None = None
        self.control: int | None = None
        self.installed = False
        self._configure_api()

    def _configure_api(self) -> None:
        self.kernel32.VirtualAllocEx.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.kernel32.VirtualAllocEx.restype = wintypes.LPVOID
        self.kernel32.VirtualFreeEx.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        self.kernel32.VirtualFreeEx.restype = wintypes.BOOL
        self.kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        self.ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
        self.ntdll.NtSuspendProcess.restype = wintypes.LONG
        self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.ntdll.NtResumeProcess.restype = wintypes.LONG

    @contextmanager
    def _suspended(self):
        status = int(self.ntdll.NtSuspendProcess(self.process.handle))
        if status < 0:
            raise RuntimeError(f"NtSuspendProcess failed: NTSTATUS {status:#010x}")
        try:
            yield
        finally:
            status = int(self.ntdll.NtResumeProcess(self.process.handle))
            if status < 0:
                raise RuntimeError(f"NtResumeProcess failed: NTSTATUS {status:#010x}")

    def _process_alive(self) -> bool:
        if not self.process.handle:
            return False
        exit_code = wintypes.DWORD()
        return bool(
            self.kernel32.GetExitCodeProcess(
                self.process.handle, ctypes.byref(exit_code)
            )
            and exit_code.value == STILL_ACTIVE
        )

    def _pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        handle = self.kernel32.OpenProcess(
            SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(
                self.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                and exit_code.value == STILL_ACTIVE
            )
        finally:
            self.kernel32.CloseHandle(handle)

    def _write(self, address: int, data: bytes, *, verify: bool = True) -> None:
        buffer = ctypes.create_string_buffer(data)
        written = ctypes.c_size_t()
        if not self.kernel32.WriteProcessMemory(
            self.process.handle,
            ctypes.c_void_p(address),
            buffer,
            len(data),
            ctypes.byref(written),
        ) or written.value != len(data):
            raise ctypes.WinError(ctypes.get_last_error())
        if verify and self.process.read(address, len(data)) != data:
            raise RuntimeError(f"input write did not verify at {address:#x}")

    def _write_code(self, address: int, data: bytes) -> None:
        old = wintypes.DWORD()
        if not self.kernel32.VirtualProtectEx(
            self.process.handle,
            ctypes.c_void_p(address),
            len(data),
            PAGE_EXECUTE_READWRITE,
            ctypes.byref(old),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self._write(address, data)
        finally:
            restored = wintypes.DWORD()
            if not self.kernel32.VirtualProtectEx(
                self.process.handle,
                ctypes.c_void_p(address),
                len(data),
                old.value,
                ctypes.byref(restored),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        if not self.kernel32.FlushInstructionCache(
            self.process.handle,
            ctypes.c_void_p(address),
            len(data),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def _discard_allocation(self) -> None:
        if self.cave and self.process.handle and self._process_alive():
            self.kernel32.VirtualFreeEx(
                self.process.handle,
                ctypes.c_void_p(self.cave),
                0,
                MEM_RELEASE,
            )
        self.cave = None
        self.control = None

    def install(self) -> None:
        if self.installed:
            return
        actual = self.process.read(HOOK_ADDRESS, HOOK_SIZE)
        if actual != HOOK_ORIGINAL:
            self._recover_verified_orphan(actual)
            actual = self.process.read(HOOK_ADDRESS, HOOK_SIZE)
            if actual != HOOK_ORIGINAL:
                raise RuntimeError("orphan recovery did not restore the input hook")
        allocation = self.kernel32.VirtualAllocEx(
            self.process.handle,
            None,
            CAVE_SIZE,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
        if not allocation:
            raise ctypes.WinError(ctypes.get_last_error())
        self.cave = int(allocation)
        self.control = self.cave + CONTROL_OFFSET
        try:
            self._write(self.cave, build_stub(self.cave))
            self._write(
                self.control,
                CONTROL_STRUCT.pack(0, os.getpid(), CONTROL_MAGIC),
            )
            hook = build_hook(self.cave)
            with self._suspended():
                if self.process.read(HOOK_ADDRESS, HOOK_SIZE) != HOOK_ORIGINAL:
                    raise RuntimeError("input call site changed before publication")
                self._write_code(HOOK_ADDRESS, hook)
                if self.process.read(HOOK_ADDRESS, HOOK_SIZE) != hook:
                    raise RuntimeError("input hook publication did not verify")
            self.installed = True
        except Exception:
            if self._process_alive() and self.process.read(
                HOOK_ADDRESS, HOOK_SIZE
            ) == build_hook(self.cave):
                with self._suspended():
                    self._write_code(HOOK_ADDRESS, HOOK_ORIGINAL)
            self._discard_allocation()
            raise

    def _recover_verified_orphan(self, hook: bytes) -> None:
        cave = call_target(HOOK_ADDRESS, hook)
        expected = build_stub(cave)
        if self.process.read(cave, len(expected)) != expected:
            raise RuntimeError(f"refusing unknown input hook target {cave:#x}")
        control = cave + CONTROL_OFFSET
        _mask, owner_pid, magic = CONTROL_STRUCT.unpack(
            self.process.read(control, CONTROL_STRUCT.size)
        )
        if magic != CONTROL_MAGIC:
            raise RuntimeError(f"refusing unknown control block {cave:#x}")
        if self._pid_alive(owner_pid):
            raise RuntimeError(
                f"TH06 input bridge is owned by active controller pid {owner_pid}"
            )
        self.cave = cave
        self.control = control
        self.installed = True
        self.close()

    def set_mask(self, mask: int) -> None:
        if not self.installed or self.control is None:
            raise RuntimeError("TH06 background input bridge is not installed")
        if mask & BUTTON_BOMB:
            raise ValueError("Bomb bit 0x02 is forbidden")
        if mask & ~ALLOWED_INPUT_MASK:
            raise ValueError(f"unsupported TH06 input bits: 0x{mask:04X}")
        if not self._process_alive():
            if mask == 0:
                return
            raise RuntimeError("TH06 process exited before input publication")
        self._write(self.control, struct.pack("<I", mask))

    def set_keys(self, names: set[str]) -> None:
        self.set_mask(keys_to_input_mask(names))

    def release_all(self) -> None:
        if self.installed and self.control is not None:
            self.set_mask(0)

    def close(self) -> None:
        if not self.installed and self.cave is None:
            return
        self.release_all()
        if self.installed and self._process_alive():
            with self._suspended():
                current = self.process.read(HOOK_ADDRESS, HOOK_SIZE)
                if current != build_hook(self.cave):
                    raise RuntimeError(
                        f"input call site changed before removal: {current.hex(' ')}"
                    )
                self._write_code(HOOK_ADDRESS, HOOK_ORIGINAL)
                if self.process.read(HOOK_ADDRESS, HOOK_SIZE) != HOOK_ORIGINAL:
                    raise RuntimeError("input hook restoration did not verify")
            self.installed = False
            # The only in-flight reference is one tiny CALL/RET; wait more than
            # a normal frame before freeing the verified allocation.
            time.sleep(0.05)
        else:
            self.installed = False
        self._discard_allocation()

    def __enter__(self) -> "BackgroundInputBridge":
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
