from scripts.export_generation6_policy import valid_canary_schema


def test_generation6_canary_schema_is_version_extensible() -> None:
    assert valid_canary_schema("autonomous-generation-6-wine-canary-v1")
    assert valid_canary_schema("autonomous-generation-6-wine-canary-v4")
    assert valid_canary_schema("autonomous-generation-6-wine-canary-v12")


def test_generation6_canary_schema_rejects_unversioned_or_foreign_values() -> None:
    assert not valid_canary_schema("autonomous-generation-6-wine-canary")
    assert not valid_canary_schema("autonomous-generation-5-wine-canary-v4")
    assert not valid_canary_schema("autonomous-generation-6-wine-canary-v4x")
    assert not valid_canary_schema(None)
