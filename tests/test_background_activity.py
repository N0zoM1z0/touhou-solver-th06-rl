import pytest

from th06_rl.th06.background_activity import _decode_activity


def test_activity_source_flag_accepts_boolean_wparam_values():
    assert _decode_activity(b"\x00\x00\x00\x00") == 0
    assert _decode_activity(b"\x01\x00\x00\x00") == 1


def test_activity_source_flag_rejects_incoherent_values():
    with pytest.raises(RuntimeError):
        _decode_activity(b"\x02\x00\x00\x00")
    with pytest.raises(RuntimeError):
        _decode_activity(b"\x00")
