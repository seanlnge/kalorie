#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-data/earnings_call_transcripts}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/sec-corpus-ex99-supplemental.json}"
DATASET_OUT="${DATASET_OUT:-artifacts/model1/datasets/synthetic-phrase-presence-2kplus.json}"
MODEL_OUT="${MODEL_OUT:-artifacts/model1/models/model1.json}"
OPTIMIZED_MODEL_OUT="${OPTIMIZED_MODEL_OUT:-artifacts/model1/models/model1-optimized.json}"
OPTIMIZATION_REPORT_OUT="${OPTIMIZATION_REPORT_OUT:-artifacts/model1/eval/model1-optimization-report.json}"
EVAL_OUT="${EVAL_OUT:-artifacts/model1/eval/historical-eval.json}"
MODEL1_EVAL_OUT="${MODEL1_EVAL_OUT:-artifacts/model1/eval/model1-holdout-eval.json}"
MARKET_CONTRACTS_PATH="${MARKET_CONTRACTS_PATH:-data/kalshi/combined-closed-mention-markets.json}"
MIN_COMPANY_ROWS="${MIN_COMPANY_ROWS:-25}"
BLEND_WEIGHT="${BLEND_WEIGHT:-0.35}"
RECORD_CONCURRENCY="${RECORD_CONCURRENCY:-100}"
OPTIMIZE_MODEL1="${OPTIMIZE_MODEL1:-1}"

TARGET_PHRASES="revenue,margin,guidance,traffic,automation,ai,tariff,inflation,demand,pricing,cloud,inventory,capex,operating margin,gross margin,free cash flow,cash flow,share repurchase,buyback,dividend,backlog,bookings,pipeline,deal activity,enterprise,consumer,advertising,subscriptions,churn,retention,headcount,hiring,restructuring,layoffs,supply chain,china,europe,regulation,compliance,litigation,cybersecurity,data center,gpu,compute,inference,training model,open source,partnership,m&a,acquisition,integration,restaurant-level margin,same restaurant sales,digital mix,new store openings,unit growth,throughput,promotions,discounting,working capital"

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

run_stage "Build synthetic dataset" \
  python -m kalorie.app.cli build-synthetic-phrase-dataset --transcript-root "$TRANSCRIPT_ROOT" --manifests "$MANIFEST_PATH" --target-phrases "$TARGET_PHRASES" --market-contracts "$MARKET_CONTRACTS_PATH" --record-concurrency "$RECORD_CONCURRENCY" --out "$DATASET_OUT"

if [[ "$OPTIMIZE_MODEL1" == "1" ]]; then
  run_stage "Train optimized model1 (Brier grid search)" \
    python -m kalorie.app.cli train-model1-optimized --examples "$DATASET_OUT" --out "$OPTIMIZED_MODEL_OUT" --report-out "$OPTIMIZATION_REPORT_OUT"
  MODEL_OUT="$OPTIMIZED_MODEL_OUT"
else
  run_stage "Train base model1" \
    python -m kalorie.app.cli train-model1 --examples "$DATASET_OUT" --out "$MODEL_OUT" --min-company-rows "$MIN_COMPANY_ROWS" --blend-weight "$BLEND_WEIGHT"
fi

run_stage "Evaluate holdout brier/logloss" \
  python -m kalorie.app.cli train-model --examples "$DATASET_OUT" --out "$EVAL_OUT"

run_stage "Evaluate model1 holdout set" \
  python -m kalorie.app.cli evaluate-model1 --model "$MODEL_OUT" --examples "$DATASET_OUT" --out "$MODEL1_EVAL_OUT"

echo "Done. Outputs:"
echo "  Dataset: $DATASET_OUT"
echo "  Model:   $MODEL_OUT"
if [[ "$OPTIMIZE_MODEL1" == "1" ]]; then
  echo "  Optimization: $OPTIMIZATION_REPORT_OUT"
fi
echo "  Eval:    $EVAL_OUT"
echo "  Model1 Eval: $MODEL1_EVAL_OUT"
FLOW_ENDED="$(date +%s)"
echo "Total elapsed: $((FLOW_ENDED - FLOW_STARTED))s"
