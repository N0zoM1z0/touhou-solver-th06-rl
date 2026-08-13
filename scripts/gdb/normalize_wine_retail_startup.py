"""GDB-side normalization for TH06 1.02h's Wine-only first-frame catch-up."""

from __future__ import annotations

import gdb


BREAKPOINT = 0x0042097E
LOOP_BACK_EDGE = 0x0042098E
LAST_FRAME_TIME = 0x006C6BF8
MAIN_MENU_STATE = 0x006DC8B0
EXPECTED_BREAKPOINT_BYTES = bytes.fromhex("dd 45 d0")
EXPECTED_BACK_EDGE_BYTES = bytes.fromhex("74 d0")


inferior = gdb.selected_inferior()


def read(address: int, size: int) -> bytes:
    return bytes(inferior.read_memory(address, size))


def write(address: int, payload: bytes) -> None:
    inferior.write_memory(address, payload)


# Wine legitimately uses SIGUSR1 while a freshly attached process is still
# starting. GDB's default stop would return from `continue` before TH06 reaches
# the authoritative timing-loop breakpoint and make a successful attach look
# like a version mismatch. Pass the signal through and keep waiting.
gdb.execute("handle SIGUSR1 nostop noprint pass", to_string=True)
stop = gdb.Breakpoint(f"*0x{BREAKPOINT:08x}", internal=False)
gdb.execute("continue", to_string=True)
stop.delete()
pc = int(gdb.parse_and_eval("$pc"))
if pc != BREAKPOINT:
    raise RuntimeError(f"TH06 did not reach timing loop: pc=0x{pc:08x}")
if read(BREAKPOINT, 3) != EXPECTED_BREAKPOINT_BYTES:
    raise RuntimeError("TH06 timing-loop breakpoint bytes do not match v1.02h")
if read(LOOP_BACK_EDGE, 2) != EXPECTED_BACK_EDGE_BYTES:
    raise RuntimeError("TH06 timing-loop back-edge bytes do not match v1.02h")
menu_state = int.from_bytes(read(MAIN_MENU_STATE, 4), "little", signed=True)
if menu_state != 1:
    raise RuntimeError(f"TH06 startup state changed before normalization: {menu_state}")

ebp = int(gdb.parse_and_eval("$ebp"))
current_time = read(ebp - 0x28, 8)
write(LAST_FRAME_TIME, current_time)
write(ebp - 0x30, bytes(8))

gdb.write(
    "TH06_RL_WINE_STARTUP normalized=1 "
    f"pc=0x{pc:08x} menu_state={menu_state}\n"
)
gdb.execute("detach")
gdb.execute("quit")
