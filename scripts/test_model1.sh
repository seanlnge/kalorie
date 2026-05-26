#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-artifacts/model1/datasets/synthetic-phrase-presence-2kplus.json}"
MODEL_PATH="${MODEL_PATH:-artifacts/model1/models/model1.json}"
FEATURES_PATH="${FEATURES_PATH:-artifacts/model1/datasets/smoke-features.json}"
PREDICTION_OUT="${PREDICTION_OUT:-artifacts/model1/predictions/smoke-predictions.json}"
COMPANY_SYMBOL="${COMPANY_SYMBOL:-CAVA}"

FLOW_STARTED="$(date +%s)"

run_stage() {
  local label="$1"
  shift
  local started
  started="$(date +%s)"
  echo "[stage] START $label"
  "$@"
  local ended
  ended="$(date +%s)"
  echo "[stage] DONE  $label ($((ended - started))s)"
}

run_stage "Run focused model tests" \
  python -m pytest \
  tests/unit/test_model1.py \
  tests/integration/test_cli_historical_training.py::test_train_and_predict_model1_cli \
  -q

run_stage "Build smoke feature vectors" \
  python -c "import json,pathlib; ds=pathlib.Path(r'$DATASET_PATH'); out=pathlib.Path(r'$FEATURES_PATH'); rows=json.loads(ds.read_text(encoding='utf-8')); picked=[]; seen=set();
for row in rows:
    phrase=row['target_phrase']
    if phrase in seen:
        continue
    seen.add(phrase)
    picked.append({'target_phrase': phrase, 'features': row['features']})
    if len(picked) >= 10:
        break
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(picked, indent=2), encoding='utf-8'); print(f'Wrote {len(picked)} smoke feature rows to {out}')"

run_stage "Generate smoke predictions" \
  python -m kalorie.app.cli predict-model1 \
  --model "$MODEL_PATH" \
  --features "$FEATURES_PATH" \
  --company-symbol "$COMPANY_SYMBOL" \
  --out "$PREDICTION_OUT"

echo "Smoke prediction output: $PREDICTION_OUT"
FLOW_ENDED="$(date +%s)"
echo "Total elapsed: $((FLOW_ENDED - FLOW_STARTED))s"
