from th06_rl.th06.controller import _next_anchor_reason


def test_stage_change_cannot_be_downgraded_to_source_context_change() -> None:
    reason = _next_anchor_reason(
        None,
        last_stage=1,
        current_stage=2,
        last_partition="stage:1:timeline",
        current_partition="stage:2:timeline",
        source_program_extended=True,
        periodic_due=True,
    )

    assert reason == "stage-root"


def test_same_stage_anchor_reasons_keep_strict_precedence() -> None:
    assert _next_anchor_reason(
        None,
        last_stage=2,
        current_stage=2,
        last_partition="stage:2:timeline",
        current_partition="stage:2:boss:mid",
        source_program_extended=True,
        periodic_due=True,
    ) == "source-context-change"
    assert _next_anchor_reason(
        None,
        last_stage=2,
        current_stage=2,
        last_partition="stage:2:timeline",
        current_partition="stage:2:timeline",
        source_program_extended=True,
        periodic_due=True,
    ) == "source-program-extension"
    assert _next_anchor_reason(
        "stage-root",
        last_stage=2,
        current_stage=2,
        last_partition=None,
        current_partition="stage:2:timeline",
        source_program_extended=False,
        periodic_due=False,
    ) == "stage-root"
