import json
from datetime import datetime
from pathlib import Path

from kalorie.workflows.models import WorkflowVerificationReport


def verify_event_pack_artifacts(pack_dir: Path) -> WorkflowVerificationReport:
    errors: list[str] = []
    warnings: list[str] = []
    for event_dir in sorted(path for path in pack_dir.iterdir() if path.is_dir()):
        contracts_path = event_dir / "contracts.json"
        snapshots_path = event_dir / "snapshots.json"
        for required in [contracts_path, snapshots_path]:
            if not required.exists():
                errors.append(f"missing_file:{required}")
                continue
            try:
                json.loads(required.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid_json:{required}:{exc}")

        contracts = _load_json_array(contracts_path)
        snapshots = _load_json_array(snapshots_path)
        manifests = _load_json_array(event_dir / "evidence-manifests.json")
        if not (event_dir / "transcript" / "transcript.txt").exists():
            errors.append(f"missing_transcript:{event_dir.name}")
        if manifests == []:
            errors.append(f"missing_evidence:{event_dir.name}")
        if contracts is not None and snapshots is not None and len(contracts) != len(snapshots):
            errors.append(f"snapshot_contract_count_mismatch:{event_dir.name}")

        if not snapshots_path.exists():
            continue
        try:
            snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(snapshots, list):
            errors.append(f"invalid_snapshots_shape:{snapshots_path}")
            continue
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                errors.append(f"invalid_snapshot_row:{snapshots_path}")
                continue
            cutoff_raw = snapshot.get("snapshot_target_time")
            candle_ts = snapshot.get("candle_end_ts")
            if cutoff_raw is None or candle_ts is None:
                warnings.append(f"incomplete_snapshot:{snapshots_path}")
                continue
            cutoff = datetime.fromisoformat(str(cutoff_raw).replace("Z", "+00:00"))
            if int(candle_ts) > int(cutoff.timestamp()):
                errors.append(
                    f"post_cutoff_snapshot:{event_dir.name}:{snapshot.get('market_id', 'unknown')}"
                )

    return WorkflowVerificationReport(ok=not errors, errors=errors, warnings=warnings)


def _load_json_array(path: Path) -> list | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, list) else None
