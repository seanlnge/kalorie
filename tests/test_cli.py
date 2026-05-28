from typer.testing import CliRunner

from kalorie2 import cli
from kalorie2.models import CollectionResult


def test_collect_cli_defaults_to_three_seeded_random_snapshot_samples(tmp_path, monkeypatch):
    captured = {}

    class FakeCollector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def collect(self):
            return CollectionResult(
                rows=[],
                skipped_markets=[],
                stats={"rows_written": 0, "skipped_count": 0},
            )

    monkeypatch.setattr(cli, "HistoricalMentionCollector", FakeCollector)

    result = CliRunner().invoke(cli.app, ["--out-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["snapshot_samples_per_market"] == 3
    assert captured["snapshot_min_hours_before_close"] == 2
    assert captured["snapshot_max_hours_before_close"] == 48
    assert captured["snapshot_sampling_seed"] == 0

