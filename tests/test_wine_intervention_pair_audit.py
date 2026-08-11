from scripts.audit_wine_intervention_pair import _event_identity, _root_hash


def test_root_hash_ignores_capture_mechanics_only() -> None:
    left = {
        "frame": 100,
        "x": 10.0,
        "rng_seed": 7,
        "raw_bullet_tails": {"codec": "bytes-base64-v1", "data": "AA=="},
        "capture_attempts": 1,
        "bullet_read_retries": 0,
    }
    right = dict(left, capture_attempts=3, bullet_read_retries=2)
    assert _root_hash(left) == _root_hash(right)
    right["rng_seed"] = 8
    assert _root_hash(left) != _root_hash(right)


def test_event_identity_decodes_balanced_pair_contract() -> None:
    identity = _event_identity({
        "policy_id": (
            "wine-one-shot-intervention-v1:pair-01:alternative:"
            "left_fast:up_fast"
        )
    })
    assert identity == {
        "pair_id": "pair-01",
        "arm": "alternative",
        "incumbent_action": "left_fast",
        "alternative_action": "up_fast",
    }
