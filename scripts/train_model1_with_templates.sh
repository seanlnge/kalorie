#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-data/earnings_call_transcripts}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/sec-corpus-ex99-supplemental.json}"
DATASET_OUT="${DATASET_OUT:-artifacts/model1/datasets/synthetic-phrase-presence-2kplus-with-templates.json}"
MODEL_OUT="${MODEL_OUT:-artifacts/model1/models/model1-with-templates.json}"
OPTIMIZED_MODEL_OUT="${OPTIMIZED_MODEL_OUT:-artifacts/model1/models/model1-optimized.json}"
OPTIMIZATION_REPORT_OUT="${OPTIMIZATION_REPORT_OUT:-artifacts/model1/eval/model1-optimization-report.json}"
COMPANY_MODEL_OUT="${COMPANY_MODEL_OUT:-artifacts/model1/models/model1-company-specific.json}"
EVAL_OUT="${EVAL_OUT:-artifacts/model1/eval/historical-eval-with-templates.json}"
MODEL1_EVAL_OUT="${MODEL1_EVAL_OUT:-artifacts/model1/eval/model1-holdout-eval.json}"
COMPANY_EVAL_EXAMPLES="${COMPANY_EVAL_EXAMPLES:-data/datasets/training/historical/cava-2026-q1-clean-examples.json}"
COMPANY_EVAL_OUT="${COMPANY_EVAL_OUT:-artifacts/model1/eval/model1-cava-eval.json}"
TEMPLATE_CATALOG_OUT="${TEMPLATE_CATALOG_OUT:-artifacts/model1/datasets/template-phrases.json}"
MARKET_CONTRACTS_PATH="${MARKET_CONTRACTS_PATH:-data/kalshi/combined-closed-mention-markets.json}"
MIN_COMPANY_ROWS="${MIN_COMPANY_ROWS:-25}"
BLEND_WEIGHT="${BLEND_WEIGHT:-0.35}"
TARGET_COMPANY_SYMBOL="${TARGET_COMPANY_SYMBOL:-}"
TEMPLATE_COMPANY_SYMBOL="${TEMPLATE_COMPANY_SYMBOL:-${TARGET_COMPANY_SYMBOL}}"
TEMPLATE_MAX_CONCURRENCY="${TEMPLATE_MAX_CONCURRENCY:-60}"
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

if [[ -n "$TEMPLATE_COMPANY_SYMBOL" ]]; then
  run_stage "Generate template catalog" \
    python -m kalorie.app.cli generate-template-phrases --manifests "$MANIFEST_PATH" --target-phrases "$TARGET_PHRASES" --company-symbol "$TEMPLATE_COMPANY_SYMBOL" --max-concurrency "$TEMPLATE_MAX_CONCURRENCY" --out "$TEMPLATE_CATALOG_OUT"
else
  run_stage "Generate template catalog" \
    python -m kalorie.app.cli generate-template-phrases --manifests "$MANIFEST_PATH" --target-phrases "$TARGET_PHRASES" --max-concurrency "$TEMPLATE_MAX_CONCURRENCY" --out "$TEMPLATE_CATALOG_OUT"
fi

run_stage "Build synthetic dataset with template features" \
  python -m kalorie.app.cli build-synthetic-phrase-dataset --transcript-root "$TRANSCRIPT_ROOT" --manifests "$MANIFEST_PATH" --target-phrases "$TARGET_PHRASES" --template-catalog "$TEMPLATE_CATALOG_OUT" --market-contracts "$MARKET_CONTRACTS_PATH" --record-concurrency "$RECORD_CONCURRENCY" --out "$DATASET_OUT"

if [[ "$OPTIMIZE_MODEL1" == "1" ]]; then
  run_stage "Train optimized model1 (Brier grid search)" \
    python -m kalorie.app.cli train-model1-optimized --examples "$DATASET_OUT" --out "$OPTIMIZED_MODEL_OUT" --report-out "$OPTIMIZATION_REPORT_OUT"
  MODEL_OUT="$OPTIMIZED_MODEL_OUT"
else
  run_stage "Train base model1" \
    python -m kalorie.app.cli train-model1 --examples "$DATASET_OUT" --out "$MODEL_OUT" --min-company-rows "$MIN_COMPANY_ROWS" --blend-weight "$BLEND_WEIGHT"
fi

if [[ -n "$TARGET_COMPANY_SYMBOL" ]]; then
  run_stage "Train company retrained model1 ($TARGET_COMPANY_SYMBOL)" \
    python -m kalorie.app.cli train-model1-company --examples "$DATASET_OUT" --company-symbol "$TARGET_COMPANY_SYMBOL" --base-model "$MODEL_OUT" --out "$COMPANY_MODEL_OUT" --min-company-rows "$MIN_COMPANY_ROWS"
fi

run_stage "Evaluate holdout brier/logloss" \
  python -m kalorie.app.cli train-model --examples "$DATASET_OUT" --out "$EVAL_OUT"

run_stage "Evaluate model1 holdout set" \
  python -m kalorie.app.cli evaluate-model1 --model "$MODEL_OUT" --examples "$DATASET_OUT" --out "$MODEL1_EVAL_OUT"

if [[ -f "$COMPANY_EVAL_EXAMPLES" ]]; then
  run_stage "Evaluate model1 on company slice" \
    python -m kalorie.app.cli evaluate-model1 --model "$MODEL_OUT" --examples "$COMPANY_EVAL_EXAMPLES" --out "$COMPANY_EVAL_OUT"
fi

echo "Done. Outputs:"
echo "  Template Catalog: $TEMPLATE_CATALOG_OUT"
echo "  Dataset:          $DATASET_OUT"
echo "  Model:            $MODEL_OUT"
if [[ "$OPTIMIZE_MODEL1" == "1" ]]; then
  echo "  Optimization:     $OPTIMIZATION_REPORT_OUT"
fi
if [[ -n "$TARGET_COMPANY_SYMBOL" ]]; then
  echo "  Company Model:    $COMPANY_MODEL_OUT"
fi
echo "  Eval:             $EVAL_OUT"
echo "  Model1 Eval:      $MODEL1_EVAL_OUT"
if [[ -f "$COMPANY_EVAL_EXAMPLES" ]]; then
  echo "  Company Eval:     $COMPANY_EVAL_OUT"
fi
FLOW_ENDED="$(date +%s)"
echo "Total elapsed: $((FLOW_ENDED - FLOW_STARTED))s"
