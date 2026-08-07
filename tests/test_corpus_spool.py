from __future__ import annotations

import gzip
import hashlib
import json

from scripts.finalize_corpus_spool import finalize_spool


def test_local_spool_is_recompressed_then_archived(tmp_path) -> None:
    spool = tmp_path / "spool"
    archive = tmp_path / "archive"
    run = spool / "run-1"
    run.mkdir(parents=True)
    payload = (b'{"frame":1}\n' * 1000)
    shard = run / "frames-000000-spool.jsonl.gz"
    with shard.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=0, mtime=0
        ) as output:
            output.write(payload)
    compressed = shard.read_bytes()
    manifest = {
        "run_id": "run-1",
        "storage_compression": "gzip-0",
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(payload),
        "archive_max_run_bytes": 1024 * 1024,
        "shards": [{
            "stream": "frames",
            "path": shard.name,
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "records": 1000,
            "first_sequence": 0,
            "last_sequence": 999,
            "uncompressed_bytes": len(payload),
            "compressed_bytes": len(compressed),
        }],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "run.json").write_text(
        json.dumps({"run_id": "run-1", "storage": {"compression": "gzip-0"}}),
        encoding="utf-8",
    )

    archived = finalize_spool(spool, archive)

    assert archived == [archive / "run-1"]
    assert not run.exists()
    final_manifest = json.loads(
        (archived[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["storage_compression"] == "gzip-3"
    final_shard = archived[0] / final_manifest["shards"][0]["path"]
    assert hashlib.sha256(final_shard.read_bytes()).hexdigest() == (
        final_manifest["shards"][0]["sha256"]
    )
    with gzip.open(final_shard, "rb") as source:
        assert source.read() == payload
