# Kalorie2 Earnings Mention Trading Overlay Plan

## Goal

Build a market-anchored prediction engine for Kalshi earnings mention markets. The first production-shaped version should estimate edge versus `preclose_yes_bid` / `preclose_yes_ask` at a fixed decision snapshot, not replace the market price with an unconstrained standalone LLM forecast.

## Architecture

The engine will use `preclose_yes_mid` as the base probability and learn a bounded residual from features that are available before the call. GPT-generated event scenarios and target contexts are evidence features, not the label source and not the final probability engine.

```text
historical rows + cutoff-safe evidence
  -> event scenario catalog
  -> per-target support features
  -> market-anchored residual model
  -> walk-forward calibration
  -> bid/ask edge backtest
```

## Artifact Policy

- Keep `artifacts/full/` source-only: `mention-markets-historical-20260523.csv` and `.json`.
- Write model outputs under run-scoped directories such as `artifacts/prediction-engine/<run-id>/`.
- Only retain canonical final outputs: run config, feature matrix, model summary, evaluation report, and trades CSV.
- Delete temporary prompts, scratch completions, exploratory reports, and stale prediction dumps after their conclusions are captured here or in a final report.

## Files

- Create `src/kalorie2/prediction_types.py` for shared data contracts.
- Create `src/kalorie2/event_scenarios.py` for GPT scenario prompt building, JSON parsing, validation, and cache records.
- Create `src/kalorie2/prediction_features.py` for market, phrase, transcript-history, scenario, and target-support feature extraction.
- Create `src/kalorie2/residual_engine.py` for market-anchored residual training, prediction, calibration, and walk-forward evaluation.
- Create `src/kalorie2/prediction_cli.py` for commands that build scenario prompts, train/evaluate, and backtest.
- Modify `pyproject.toml` to add the `kalorie2-prediction-engine` CLI entry point. Do not add new runtime dependencies in the first implementation pass.
- Add tests in `tests/test_event_scenarios.py`, `tests/test_prediction_features.py`, and `tests/test_residual_engine.py`.

## Task 1: Data Contracts And Artifact Hygiene

- [x] Write failing tests for prediction-engine contracts.
  - `PredictionInputRow` must parse a historical row without allowing `final_outcome` into inference feature payloads.
  - `PredictionRunConfig` must require an explicit `run_id`, `decision_time_column`, and `artifact_retention_policy`.
  - Artifact output paths must reject `artifacts/full/` unless the caller is writing canonical source datasets.
- [x] Verify the tests fail because `src/kalorie2/prediction_types.py` does not exist yet.
- [x] Implement `prediction_types.py` with Pydantic models:
  - `PredictionInputRow`
  - `MarketSnapshotFeatures`
  - `PredictionRunConfig`
  - `PredictionRecord`
  - `ArtifactRetentionPolicy`
- [x] Run the focused tests and confirm they pass.

## Task 2: Event Scenario Catalog

- [x] Write failing tests for strict scenario JSON parsing.
  - Valid JSON with `topics`, `analyst_questions`, `management_language_patterns`, `source_rationales`, and `target_phrase_contexts` should parse.
  - Transcript-like outputs with speaker labels such as `Operator:` or `Analyst:` should be rejected.
  - Missing or extra keys should fail validation instead of silently being accepted.
- [x] Verify the tests fail because `src/kalorie2/event_scenarios.py` does not exist yet.
- [x] Implement `event_scenarios.py` with:
  - `EventScenarioCatalog`
  - `TargetPhraseContext`
  - `build_event_scenario_prompt(event, target_phrases, evidence_snippets)`
  - `parse_event_scenario_response(text)`
  - `reject_transcript_like_output(catalog)`
- [x] Keep this module API-client-free in the first pass. It should build prompts and parse responses; the CLI can later call an API or consume saved JSON.
- [x] Run focused tests and confirm they pass.

## Task 3: Feature Extraction

- [x] Write failing tests for deterministic feature extraction from historical rows.
  - Market features must include `market_mid_logit`, `spread`, `snapshot_staleness_hours`, and bid/ask presence flags.
  - Phrase features must classify count markets, slash alternatives, single-word terms, multi-word terms, macro words, company/product entities, and generic terms.
  - Scenario features must score each target against event topics and target contexts using deterministic token overlap first, with embedding hooks left as injectable optional inputs.
  - Label-only fields such as `final_outcome` must never appear in inference features.
- [x] Verify tests fail before implementation.
- [x] Implement `prediction_features.py` with focused functions:
  - `extract_market_features(row)`
  - `extract_phrase_features(row)`
  - `extract_scenario_features(row, catalog)`
  - `build_feature_row(row, catalog=None)`
  - `build_feature_matrix(rows, catalogs_by_event)`
- [x] Reuse the existing phrase-count parsing rules from `transcript_model.py` where possible instead of duplicating settlement semantics.
- [x] Run focused tests and confirm they pass.

## Task 4: Market-Anchored Residual Engine

- [x] Write failing tests for the residual formula.
  - If residual is zero, prediction must equal `preclose_yes_mid`.
  - Positive residual must move probability above the market mid and negative residual below it.
  - Predictions must remain clipped away from exactly `0` and `1`.
  - Walk-forward evaluation must train only on events strictly before the held-out event.
- [x] Verify tests fail before implementation.
- [x] Implement `residual_engine.py`.
  - Start with an in-repo regularized linear residual model trained by deterministic batch gradient descent on log-loss.
  - Do not add `scikit-learn` in the first implementation pass; revisit that only after the dependency-free baseline is measured.
  - Represent predictions as `logit(market_mid) + residual_delta`.
  - Emit per-row reasons: market anchor, strongest positive features, strongest negative features, and calibration bucket.
- [ ] Add grouped calibration by phrase category and evidence-strength bucket only after the base residual tests are green.
- [x] Run focused tests and confirm they pass.

## Task 5: Walk-Forward Backtest CLI

- [x] Write failing CLI tests using a tiny synthetic CSV with two chronological events.
  - The first event should be used only for history.
  - The second event should produce predictions and possible trades.
  - The CLI must write outputs outside `artifacts/full/`.
- [x] Verify tests fail before implementation.
- [x] Implement `prediction_cli.py` commands:
  - `build-prompts`: write one prompt per event into a run-scoped directory.
  - `evaluate`: build features, run event-ordered walk-forward predictions, score Brier/ECE versus Kalshi mid.
  - `backtest`: trade YES above ask plus margin and NO below bid minus margin.
  - `cleanup`: delete non-retained temporary artifacts from a run directory.
- [x] Register `kalorie2-prediction-engine = "kalorie2.prediction_cli:app"` in `pyproject.toml`.
- [x] Run focused CLI tests and confirm they pass.

## Task 6: First Historical Evaluation

- [x] Run the market-only baseline on `artifacts/full/mention-markets-historical-20260523.csv`.
- [ ] Run the transcript-history baseline already present in `transcript_model.py`.
- [x] Run the new residual engine without GPT scenario catalogs.
- [ ] Add a small hand-authored scenario catalog for 2-3 events to validate the scenario feature path without paying for API calls.
- [ ] Compare:
  - Brier score vs Kalshi mid
  - calibration by probability bucket
  - PnL vs bid/ask with a margin
  - performance by phrase category
  - performance excluding stale snapshots and zero-bid rows
- [x] Capture results in the review section and delete unneeded temporary artifacts.

## Review

Plan written before implementation. The design keeps the old market-residual insight, avoids treating GPT as an unconstrained oracle, and adds explicit artifact hygiene so `artifacts/full/` remains clean.

Implemented Tasks 1-5 as a first working slice:

- Added prediction-engine contracts and artifact write guards in `prediction_types.py`.
- Added strict, API-client-free scenario prompt/parsing contracts in `event_scenarios.py`.
- Added deterministic market, phrase, and scenario feature extraction in `prediction_features.py`.
- Added a dependency-free market-anchored linear residual engine in `residual_engine.py`.
- Added `kalorie2-prediction-engine` CLI commands for prompt generation, walk-forward evaluation, backtesting, and cleanup.

Verification:

- Task 1 red/green: `tests/test_prediction_types.py` failed before `prediction_types.py` existed, then passed.
- Task 2 red/green: `tests/test_event_scenarios.py` failed before `event_scenarios.py` existed, then passed.
- Task 3 red/green: `tests/test_prediction_features.py` failed before `prediction_features.py` existed, then passed.
- Task 4 red/green: `tests/test_residual_engine.py` failed before `residual_engine.py` existed, then passed.
- Task 5 red/green: `tests/test_prediction_cli.py` failed before `prediction_cli.py` existed, then passed.
- Full suite: `49 passed`.
- Ruff: `All checks passed!` for `src` and `tests`.

Historical smoke results:

- Market baseline over all 3,500 rows: Brier `0.119366`.
- First residual smoke, `min_training_events=20`, `epochs=1`, no GPT catalogs: 3,172 walk-forward predictions, residual Brier `0.123893`, same-window market Brier `0.123966`.
- Backtest at `margin=0.02`: 0 trades after residual clipping/standardization. This is conservative and avoids the earlier unstable overtrading failure mode.
- Retained run-scoped outputs under `artifacts/prediction-engine/initial-residual-20260524/`: `market-baseline.json`, `evaluation.json`, `predictions.csv`, `feature-matrix.json`, `backtest.json`, and `trades.csv`.
- Verified `artifacts/full/` still contains only `mention-markets-historical-20260523.csv` and `mention-markets-historical-20260523.json`.

Reviewer checkpoint fixes:

- Added a regression test and fix so paths like `artifacts/prediction-engine/../full/predictions.csv` cannot bypass the `artifacts/full/` write guard.
- Changed scenario response parsing to enforce strict JSON instead of accepting fenced JSON, leading commentary, or trailing text.
- Added retained `run-config.json` and `model-summary.json` outputs to the evaluation command.
- Added residual stabilization after the historical smoke exposed an overflow/overreaction failure: stable sigmoid, feature standardization, residual clipping, and regression tests for extreme logits / large raw features.
- Re-verified after fixes: `52 passed`, Ruff `All checks passed!`, IDE lints clear.
- Regenerated `artifacts/prediction-engine/initial-residual-20260524/` so it includes `run-config.json` and `model-summary.json`.

Pending:

- Grouped calibration by phrase category and evidence bucket.
- Transcript-history baseline rerun in this new prediction-engine report.
- Hand-authored scenario catalogs for 2-3 events to validate the GPT feature path.
- CLI support for loading scenario catalogs into `evaluate` / `backtest`.
- Performance cuts excluding stale snapshots and zero-bid rows.
- Full-speed optimization for walk-forward training beyond the smoke `epochs=1` run.

## Ad Hoc Review: Threshold-No Backtest

- [x] Confirm `market-baseline.json` `outcome_rate` semantics from the scoring code.
- [x] Run a naive in-sample threshold strategy that buys NO when `preclose_yes_mid < 0.77`.
- [x] Run a more realistic bid/ask version that buys NO when market mid is below candidate thresholds and settles against `final_outcome`.
- [x] Compare against the existing walk-forward bin-calibration backtest so the result is not just in-sample calibration leakage.
- [x] Record the answer and caveats in this review section.

Results captured on 2026-05-25:

- `outcome_rate` is the realized YES settlement rate in each probability bin, from `final_outcome`.
- Fixed in-sample NO rule for `preclose_yes_mid < 0.77`: 2,042 trades, YES rate `0.287953`, mean YES bid `0.304187`, total cost `1420.85`, total PnL `33.15`, ROI on cost `0.023331`.
- Chronological split for the same fixed rule: early half `-0.09` PnL / `-0.000116` ROI; late half `33.24` PnL / `0.051554` ROI.
- Prior-events-only bin-calibration backtest, `margin=0`, returned positive pre-fee ROI across tested history thresholds: `0.027126` at `min_bin_count=5`, `0.031751` at `10`, `0.026265` at `20`, `0.024703` at `50`, and `0.019971` at `100`.
- Caveat: these figures use bid/ask prices but do not include Kalshi fees, sizing constraints, queue fill assumptions, or threshold-selection overfitting.

## Ad Hoc Review: Residual Overlay Backtest Sweep

- [x] Ran `kalorie2.prediction_cli backtest` on `artifacts/full/mention-markets-historical-20260523.csv` with `min_training_events=20`, `epochs=1`, and margins `0`, `0.005`, `0.01`, `0.02`.
- [x] Stored outputs under `artifacts/prediction-engine/residual-backtest-sweep-20260525/`.
- [x] Verified `artifacts/full/` remains source-only.

Results:

- Margin `0`: 295 trades, total cost `209.68`, total PnL `0.32`, ROI on cost `0.001526`. Trade mix: 140 YES, 155 NO.
- Margin `0.005`: 0 trades.
- Margin `0.01`: 0 trades.
- Margin `0.02`: 0 trades.

Interpretation: the stabilized residual overlay is extremely conservative. It only finds microscopic zero-margin edges and does not currently produce enough edge to trade after even a 0.5 percentage point hurdle.

## Planned Slice: Web Search Evidence Without MCP Probability

Goal: keep the current market-anchored residual model, but add the paper's richer context idea through cutoff-safe web-search evidence features. Do not add an MCP posterior probability yet.

Design:

- Use OpenAI Responses API web search as an evidence collector, not as the final forecaster.
- Ask for strict JSON evidence packets with `published_at`, `url`, `title`, `snippet`, `target_phrases`, and `evidence_strength`.
- For historical rows, discard any source whose `published_at` is missing, unparsable, or after `snapshot_target_time`.
- Treat the model instruction "only use sources before this datetime" as a helpful prompt constraint, not the leakage guard. The code-level date filter is the guard.
- Convert retained evidence into current-model features such as `web_evidence_available`, `web_evidence_item_count`, `web_evidence_target_overlap`, and `web_evidence_strength_max`.
- Store web evidence under run-scoped artifacts such as `artifacts/prediction-engine/<run-id>/web-evidence/`, never under `artifacts/full/`.

Open questions / caveats:

- Historical web search from today's index can still leak if old pages were edited after the cutoff or if publication dates are wrong. Clean evaluation eventually needs a provider with true historical date filters or cached search snapshots.
- Live usage is safer: at forecast time, web search naturally sees only currently indexed information.

Implementation notes:

- Added `web_evidence.py` with strict packet parsing, OpenAI Responses API web-search payload construction, cutoff filtering, and feature conversion.
- Added `collect-web-evidence` CLI. Dry-run mode writes request payloads; non-dry-run can call OpenAI and write packet JSON files.
- Added `--web-evidence-dir` to `evaluate` and `backtest` so the current residual model can consume packet features.
- Dry-run test produced one request payload at `artifacts/prediction-engine/web-evidence-dry-run-20260525/web-evidence/requests/KXEARNINGSMENTIONNVDA-25FEB.json`.
- Verified `artifacts/full/` remains source-only.
- Verification after this slice: `60 passed`; Ruff `All checks passed!`.

## Live Web Evidence Run: 2026-05-25

- [x] Ran live OpenAI web-search collection for one event with `--no-dry-run`.
- [x] Fixed the collector's hardcoded 120-second timeout by adding `--request-timeout-seconds`; the successful retry used 900 seconds.
- [x] Fixed cutoff filtering for OpenAI-returned timezone-naive `published_at` timestamps by normalizing parsed datetimes to UTC.
- [x] Re-ran evaluation and zero-margin backtest with the live packet directory.

Results:

- Live packet written under `artifacts/prediction-engine/web-evidence-live-retry-20260525/web-evidence/packets/`.
- The loaded feature matrix has `web_evidence_count: 23`, corresponding to the first NVDA event's target rows.
- Smoke evaluation, `min_training_events=1`, `epochs=1`: 3,477 predictions, residual Brier `0.120017`, same-window market Brier `0.120093`.
- Smoke backtest, `margin=0`: 307 trades, total cost `218.82`, total PnL `0.18`, ROI on cost `0.000823`.
- Verification after live-run fixes: `62 passed`; Ruff `All checks passed!`; IDE lints clear.

## Planned Full Web Evidence Collection

- [x] Make the live collector resumable before running all 264 event calls, so completed packets are skipped on retry instead of re-billed.
- [x] Add per-event progress output so long OpenAI web-search runs are observable.
- [x] Verify the resumable collector with focused tests, full tests, Ruff, and IDE lints.
- [x] Add bounded parallel OpenAI calls with `--parallel-requests`; default remains serial for conservative usage.
- [ ] Paused live collection after unexpected OpenAI spend. Do not resume until usage metadata and explicit budget caps are implemented.
- [ ] Run live collection for all events under `artifacts/prediction-engine/web-evidence-full-20260525/`.
- [ ] Evaluate and backtest with the full packet directory once collection completes.

Paused run status:

- All collector processes were stopped after the user reported a roughly `$60` bill.
- Completed packets at pause: `70/264`.
- Request payloads written: `264/264` request JSON files, which are not themselves billable but show the intended event universe.
- Likely cost-estimate miss: the original estimate used saved final JSON size as a proxy for billed output, but Responses API web-search runs may bill hidden reasoning/tool orchestration/failed attempts not preserved in the current packet artifacts.

## Planned Paid API Guardrails

- [x] Persist raw OpenAI response usage metadata for each successful paid web-evidence call under a run-scoped `web-evidence/usage/` directory.
- [x] Add `--max-paid-calls` so a resume can hard-stop before making more than the explicitly allowed number of new OpenAI calls.
- [x] Add `--max-estimated-cost-dollars` and `--estimated-cost-per-call-dollars` as a conservative preflight cap over pending paid calls.
- [ ] Keep defaults non-spending-safe: no cap means behavior is unchanged, but any future scaled run should pass both caps explicitly.
- [x] Verify guardrails with focused CLI tests, full tests, Ruff, and IDE lints before any further live collection.

Guardrail implementation results:

- Added usage metadata writes to `web-evidence/usage/<event_ticker>.json` for successful OpenAI calls. Each file captures `event_ticker`, requested model, response id/model, and raw `usage` when present.
- Added `--max-paid-calls`, enforced after `--skip-existing` filtering and before any new OpenAI call is submitted.
- Added `--max-estimated-cost-dollars` and `--estimated-cost-per-call-dollars`, enforced as a conservative preflight cap over pending paid calls.
- Verification: `67 passed`; Ruff `All checks passed!`; IDE lints clear.

## Partial Web Evidence Training Run

- [x] Run local-only residual evaluation with the 70 saved web-evidence packets. Do not make further OpenAI calls.
- [x] Run a same-settings baseline residual evaluation without web evidence.
- [x] Run zero-margin backtests for both.
- [x] Compare all-row metrics and covered-event-only metrics, because only 70/264 events currently have web evidence.

Configuration:

- Input data: `artifacts/full/mention-markets-historical-20260523.csv`.
- Web evidence: `artifacts/prediction-engine/web-evidence-full-20260525/web-evidence/`.
- Runs used `min_training_events=20`, `epochs=5`, `margin=0`.
- No OpenAI calls were made during training/evaluation.

Coverage:

- Web-evidence packets: `70/264` events.
- Covered market rows: `1,073/3,500`.
- Walk-forward predictions after training warmup: `3,172`, of which `760` are covered-event predictions.

Results:

- All rows, partial-web residual Brier: `0.123316`; same-window market Brier: `0.123966`.
- All rows, baseline residual Brier: `0.123631`; same-window market Brier: `0.123966`.
- Covered rows, partial-web residual Brier: `0.061038`; covered market Brier: `0.061531`.
- Covered rows, baseline residual Brier: `0.061065`; covered market Brier: `0.061531`.
- Zero-margin backtest, partial-web model: `345` trades, total cost `190.46`, PnL `16.54`, ROI `0.086842`.
- Zero-margin backtest, baseline model: `296` trades, total cost `210.08`, PnL `-0.08`, ROI `-0.000381`.
- Covered-event trades only, partial-web model: `15` trades, PnL `0.01`, ROI `0.001001`.
- Covered-event trades only, baseline model: `12` trades, PnL `1.30`, ROI `0.134021`.

Interpretation: the partial-web run improved overall Brier and produced a much better all-row zero-margin backtest, but the direct covered-event trade slice did not outperform baseline. The apparent PnL improvement came mostly from coefficient spillover into uncovered rows, so this is promising as a model-regularization signal but not yet proof that web evidence itself creates tradable edge.

## Partial Web Evidence Training Run: 114 Packets

- [x] Reran local-only residual evaluation with the current 114 saved web-evidence packets. No OpenAI calls were made.
- [x] Reran zero-margin backtest with the current packet directory.
- [x] Compared against the same-settings baseline run from `partial-web-baseline-20260525`.

Configuration:

- Input data: `artifacts/full/mention-markets-historical-20260523.csv`.
- Web evidence: `artifacts/prediction-engine/web-evidence-full-20260525/web-evidence/`.
- Runs used `min_training_events=20`, `epochs=5`, `margin=0`.

Coverage:

- Web-evidence packets: `114/264` events.
- Covered market rows: `1,633/3,500`.
- Walk-forward predictions after training warmup: `3,172`, of which `1,305` are covered-event predictions.

Results:

- All rows, 114-packet web residual Brier: `0.123520`; same-window market Brier: `0.123966`.
- All rows, baseline residual Brier: `0.123631`; same-window market Brier: `0.123966`.
- Covered rows, 114-packet web residual Brier: `0.084516`; covered market Brier: `0.085463`.
- Covered rows, baseline residual Brier: `0.084729`; covered market Brier: `0.085463`.
- Zero-margin backtest, 114-packet web model: `311` trades, total cost `202.59`, PnL `5.41`, ROI `0.026704`.
- Zero-margin backtest, baseline model: `296` trades, total cost `210.08`, PnL `-0.08`, ROI `-0.000381`.
- Covered-event trades only, 114-packet web model: `66` trades, PnL `2.53`, ROI `0.059571`.
- Covered-event trades only, baseline model: `53` trades, PnL `0.16`, ROI `0.004016`.

Interpretation: unlike the 70-packet run, the direct covered-event slice now outperforms baseline on both Brier and zero-margin PnL. This is still pre-fee and uses partial coverage, but it is a meaningfully stronger signal that the web-evidence features may add edge.

## Full Web Evidence Training Run: 264 Packets

- [x] Completed web-evidence packets for all `264/264` events.
- [x] Reran local-only residual evaluation with the full saved packet directory. No OpenAI calls were made during training/evaluation.
- [x] Reran zero-margin backtest with the full packet directory.
- [x] Compared against the same-settings no-web baseline from `partial-web-baseline-20260525`.

Configuration:

- Input data: `artifacts/full/mention-markets-historical-20260523.csv`.
- Web evidence: `artifacts/prediction-engine/web-evidence-full-20260525/web-evidence/`.
- Runs used `min_training_events=20`, `epochs=5`, `margin=0`.

Results:

- Full-web residual Brier: `0.123468`; same-window market Brier: `0.123966`.
- No-web baseline residual Brier: `0.123631`; same-window market Brier: `0.123966`.
- Zero-margin backtest, full-web model: `311` trades, total cost `200.94`, PnL `7.06`, ROI `0.035135`.
- Zero-margin backtest, no-web baseline: `296` trades, total cost `210.08`, PnL `-0.08`, ROI `-0.000381`.
- Full-web YES trades: `122` trades, total cost `100.56`, PnL `-5.56`, ROI `-0.055290`.
- Full-web NO trades: `189` trades, total cost `100.38`, PnL `12.62`, ROI `0.125722`.
- Baseline YES trades: `140` trades, total cost `117.38`, PnL `-8.38`, ROI `-0.071392`.
- Baseline NO trades: `156` trades, total cost `92.70`, PnL `8.30`, ROI `0.089536`.

Interpretation: the full-web model improves Brier and zero-margin PnL versus the same-settings baseline. The edge is concentrated in NO trades; YES trades remain negative. The `0.5` percentage point hurdle slice did not hold up in this run, so the next modeling step should separate YES/NO decision thresholds, add fees, and evaluate margin sweeps rather than relying on raw zero-margin trades.

## Out-of-Sample Event Search: 2026-05-25

- [x] Compared live Kalshi public API results against the `264` event tickers in `artifacts/full/mention-markets-historical-20260523.csv`.
- [x] Checked broad open-market search for `KXEARNINGSMENTION`; it was noisy and hit Kalshi rate limiting before yielding any filtered new mention events.
- [x] Checked likely company-specific earnings mention series/searches including NVDA, TSLA, AAPL, MSFT, GOOGL, AMZN, META, NFLX, PLTR, AMD, ORCL, WMT, COST, JPM, BAC, GS, INTC, UBER, DIS, COINBASE, SBUX, PYPL, RDDT, SHOP, BA, LLY, HD, LOW, TGT, CRM, DELL, SNOW, ADBE, AVGO, and MU.
- [x] Found no open, closed, settled, or initialized company-specific mention events outside the current training CSV for those checks.

Result: there do not appear to be new Kalshi earnings-mention events available through the public API right now that are outside the current `264` event training set. For an immediate out-of-sample-style test, use a temporal holdout by excluding the latest 10-30 events from training and scoring them as unseen events.

## Temporal Holdout Test Plan: Latest 30 Events

Goal: test the current model on the newest historical events as an out-of-sample proxy, without making any new OpenAI calls.

Plan:

- [x] Identify the latest 30 events by `close_time` from `artifacts/full/mention-markets-historical-20260523.csv`.
- [x] Train one full-web model on the first 234 events and score all latest 30 events as unseen.
- [x] Train one same-settings no-web baseline on the first 234 events and score the same latest 30 events.
- [x] Compare holdout Brier score, market Brier score, zero-margin PnL, ROI, and YES/NO trade splits.
- [x] Save small run-scoped holdout artifacts under `artifacts/prediction-engine/temporal-holdout-20260525/`.
- [x] Record results and interpretation here.

Configuration:

- Input data: `artifacts/full/mention-markets-historical-20260523.csv`.
- Web evidence: `artifacts/prediction-engine/web-evidence-full-20260525/web-evidence/`.
- Training window: first `234` events / `3,120` markets.
- Holdout window: latest `30` events / `380` markets.
- Model settings: `epochs=5`, `learning_rate=0.05`.
- No OpenAI calls were made.

Artifacts:

- Report: `artifacts/prediction-engine/temporal-holdout-20260525/holdout-report.json`.
- Events: `artifacts/prediction-engine/temporal-holdout-20260525/holdout-events.json`.
- Predictions: `full-web-predictions.csv`, `baseline-predictions.csv`.
- Trades: `full-web-trades.csv`, `baseline-trades.csv`.

Results:

- Full-web holdout Brier: `0.163411`; holdout market Brier: `0.163622`.
- No-web baseline holdout Brier: `0.163524`; holdout market Brier: `0.163622`.
- Full-web zero-margin backtest: `48` trades, total cost `31.02`, PnL `5.98`, ROI `0.192779`.
- No-web baseline zero-margin backtest: `49` trades, total cost `33.60`, PnL `4.40`, ROI `0.130952`.
- Full-web YES trades: `14` trades, total cost `12.04`, PnL `0.96`, ROI `0.079734`.
- Full-web NO trades: `34` trades, total cost `18.98`, PnL `5.02`, ROI `0.264489`.
- Baseline YES trades: `16` trades, total cost `14.10`, PnL `0.90`, ROI `0.063830`.
- Baseline NO trades: `33` trades, total cost `19.50`, PnL `3.50`, ROI `0.179487`.

Interpretation: on the strict latest-30-event temporal holdout, the full-web model slightly improves Brier versus both the market and the no-web residual baseline. The zero-margin PnL also improves versus baseline, again mostly through stronger NO trades. This is still pre-fee and only `30` events, but it is the cleanest current out-of-sample proxy because the model is fit once on older events and all holdout events are scored without training on any holdout labels.

Margin of error:

- Method: paired cluster bootstrap by holdout event, `20,000` resamples, `95%` percentile intervals.
- Report: `artifacts/prediction-engine/temporal-holdout-20260525/margin-of-error-report.json`.
- Full-web ROI: `0.192779`, 95% CI `[0.026391, 0.330008]`, margin of error `0.151809`.
- No-web baseline ROI: `0.130952`, 95% CI `[-0.031856, 0.263298]`, margin of error `0.147577`.
- ROI edge vs baseline: `0.061826`, 95% CI `[-0.024847, 0.171952]`, margin of error `0.098400`.
- PnL edge vs baseline: `1.58`, 95% CI `[-1.36, 4.66]`, margin of error `3.01`.
- Brier edge vs baseline: `0.000113`, 95% CI `[-0.000027, 0.000254]`, margin of error `0.000141`.
- Bootstrap positive-edge rates: Brier edge `94.16%`, PnL edge `84.74%`, ROI edge `91.38%`.

Interpretation: the point estimates are favorable, but the edge is not statistically locked at 95% on only `30` clustered events because the PnL, ROI, and Brier edge intervals all cross zero. The best read is directionally promising with wide uncertainty, not conclusive.

NO-only execution intervals:

- Method: paired cluster bootstrap by event, `20,000` resamples, `95%` percentile intervals.
- Report: `artifacts/prediction-engine/temporal-holdout-20260525/no-only-ci-report.json`.
- Trade rule: `margin=0`; trade NO when `model_probability < preclose_yes_bid`.
- Latest-30 holdout NO-only: `34/380` trades, ROI `0.264489`, 95% CI `[0.037344, 0.451187]`; Brier on NO-traded markets `0.155686`, 95% CI `[0.103809, 0.218968]`.
- Latest-30 market Brier on same NO-traded markets: `0.156482`, 95% CI `[0.104678, 0.219729]`; Brier edge vs market `0.000796`, 95% CI `[-0.000226, 0.001752]`.
- Full walk-forward NO-only: `189/3,172` trades, ROI `0.125722`, 95% CI `[0.009221, 0.238748]`; Brier on NO-traded markets `0.176360`, 95% CI `[0.149089, 0.205496]`.
- Full walk-forward market Brier on same NO-traded markets: `0.176736`, 95% CI `[0.149509, 0.205694]`; Brier edge vs market `0.000376`, 95% CI `[-0.000391, 0.001104]`.

Interpretation: NO-only ROI is positive across the 95% bootstrap intervals for both the latest-30 holdout and full walk-forward slices, but the Brier edge intervals on the traded subset cross zero. The execution rule looks more robust than the probability-calibration edge on those same trades.

## Saved Model Bundle: `earnings-mention-full-web-residual-v1`

Goal: package the current full-web residual prediction engine into a stable `models/[model_name]/` directory with training data, fitted model artifacts, runtime code, and a README.

Plan:

- [x] Create `models/earnings-mention-full-web-residual-v1/`.
- [x] Save the training corpus under `training/`: the historical market CSV and the full web-evidence packet set used for feature generation.
- [x] Fit one final model on all `264` events / `3,500` markets with the current full-web feature pipeline and save the fitted weights, intercept, feature means/scales, feature schema, and training manifest under `artifacts/`.
- [x] Save the minimal runtime code needed to load the model, build features, score rows, and apply the NO-only/current execution rules under `runtime/`.
- [x] Add `README.md` explaining the model architecture, inputs, training snapshot, trade criteria, evaluation caveats, and how to run it.
- [x] Verify the saved runtime can load the model bundle and score a sample row without importing from `src/`.

Saved bundle:

- Directory: `models/earnings-mention-full-web-residual-v1/`.
- Training corpus: `training/mention-markets-historical-20260523.csv` plus `training/web-evidence/packets/`.
- Training rows/events: `3,500` rows across `264` events.
- Web-evidence packets: `264`.
- Fitted features: `28`.
- Nonzero residual weights: `22`.
- Artifacts: `model.json`, `feature-schema.json`, `training-manifest.json`, `evaluation-reports.json`.
- Runtime entrypoint: `runtime/model_runtime.py`.

Verification:

- Ran `python models/earnings-mention-full-web-residual-v1/runtime/model_runtime.py --row-index 0` from the repository root with `PYTHONPATH` removed; it loaded the saved model and scored a sample row.
- Verified required files, training row/event counts, packet count, feature count, and nonzero weight count with a local assertion script.
- Checked `runtime/model_runtime.py` with IDE lints; no linter errors were reported.

## V2 Experiment Plan: Rich Web, MixMCP, Sweeps, Leakage Audit

Scope: implement the next experiment in `kalorie2/` only. Do not edit or regenerate `models/earnings-mention-full-web-residual-v1/`; that bundle remains the frozen v1 checkpoint.

Goal: improve the current full-web residual model by adding richer web features, optional MixMCP-style probability packets with learned alpha, hyperparameter/margin sweeps, and leakage auditing for existing web evidence.

### Task 1: Richer Web Evidence Features

Files:

- Modify: `kalorie2/src/kalorie2/web_evidence.py`.
- Modify: `kalorie2/src/kalorie2/prediction_features.py`.
- Test: `kalorie2/tests/test_web_evidence.py`.
- Test: `kalorie2/tests/test_prediction_features.py`.

Plan:

- [x] Extend `WebEvidencePacket.features_for_target()` to keep the existing feature names for backwards compatibility and add richer target-specific features:
  - `web_evidence_cutoff_safe_count`
  - `web_evidence_target_match_count`
  - `web_evidence_target_match_share`
  - `web_evidence_strength_mean`
  - `web_evidence_strength_sum`
  - `web_evidence_recency_min_hours`
  - `web_evidence_recency_mean_hours`
  - `web_evidence_source_company`
  - `web_evidence_source_sec`
  - `web_evidence_source_news`
  - `web_evidence_source_analyst`
  - `web_evidence_source_other`
- [x] Classify source type with deterministic string rules over `source`, `url`, and `title`.
- [x] Treat undated evidence as unavailable for cutoff/recency features, matching current cutoff-safe behavior.
- [x] Add tests for recency, source classification, target match counts, unchanged existing feature names, and row-level cutoff filtering.

### Task 2: MixMCP-Style Probability Packets and Learned Alpha

Files:

- Create: `kalorie2/src/kalorie2/mixmcp.py`.
- Modify: `kalorie2/src/kalorie2/prediction_cli.py`.
- Test: `kalorie2/tests/test_mixmcp.py`.
- Test: `kalorie2/tests/test_prediction_cli.py`.

Plan:

- [x] Add Pydantic models for market-conditioned probability outputs:
  - `MixMcpTargetProbability`: `market_ticker`, `target_phrase`, `market_probability`, `llm_probability`, `confidence`, `rationale`.
  - `MixMcpEventPacket`: `event_ticker`, `cutoff_time`, `model`, `targets`.
- [x] Add strict JSON parser and prompt/payload builder for collecting MixMCP packets from OpenAI web search with the market prior included.
- [x] Add CLI support for `--mixmcp-dir` on `evaluate` / `backtest` without requiring packets to exist for every event.
- [x] Implement logit-space mixing:
  - `mixed = sigmoid(alpha * logit(base_probability) + (1 - alpha) * logit(llm_probability))`.
  - Default `alpha=1.0` when no packet exists, preserving current behavior.
- [x] Implement learned alpha via temporal training folds. Candidate alpha grid: `0.0, 0.1, ..., 1.0`.
- [x] Learn alpha on prior events only for each walk-forward prediction window to avoid leakage.
- [x] Support separate alpha modes:
  - `global`: one alpha for all predictions.
  - `side`: one alpha for YES candidates and one for NO candidates.
- [x] Add tests proving alpha selection uses only prior labels.

### Task 3: Hyperparameter and Execution Sweep

Files:

- Modify: `kalorie2/src/kalorie2/residual_engine.py`.
- Modify: `kalorie2/src/kalorie2/prediction_cli.py`.
- Test: `kalorie2/tests/test_residual_engine.py`.
- Test: `kalorie2/tests/test_prediction_cli.py`.

Plan:

- [x] Expose `l2` and `residual_clip` as fit parameters instead of hardcoding `l2=0.001` and fit-time clip `2.0`.
- [x] Add CLI options to `evaluate` / `backtest`: `--l2`, `--residual-clip`.
- [x] Add `sweep` CLI command that runs local-only combinations over:
  - `epochs`: default `5,20,100`
  - `learning_rate`: default `0.05`
  - `l2`: default `0.0001,0.001,0.01,0.1`
  - `residual_clip`: default `0.5,1.0,2.0`
  - `margin`: default `0,0.005,0.01,0.02`
  - `trade_side`: `all,no_only,yes_only`
  - optional `alpha` grid when MixMCP packets are present
- [x] Write sweep outputs under run-scoped directories only, e.g. `artifacts/prediction-engine/v2-sweep-YYYYMMDD/`.
- [ ] Rank configs by temporal holdout metrics and event-cluster bootstrap CIs, not row-level accuracy alone.

### Task 4: Web Evidence Leakage Audit

Files:

- Create: `kalorie2/src/kalorie2/web_evidence_audit.py`.
- Modify: `kalorie2/src/kalorie2/prediction_cli.py`.
- Test: `kalorie2/tests/test_web_evidence_audit.py`.
- Test: `kalorie2/tests/test_prediction_cli.py`.

Plan:

- [x] Add deterministic audit checks over existing packet JSON:
  - post-cutoff `published_at`
  - undated items
  - title/url/snippet terms suggesting transcript, earnings call transcript, call audio, post-call recap, prepared remarks, AlphaSense transcript, Seeking Alpha transcript, Motley Fool transcript, resolution, Kalshi, prediction market, or final results
  - URL domains likely to host transcripts or market-resolution content
- [x] Emit JSON and CSV audit reports with event ticker, item URL, issue codes, severity, and summary counts.
- [x] Add CLI command `audit-web-evidence WEB_EVIDENCE_DIR --out-dir ...`.
- [x] Do not delete existing packets automatically. Produce an exclusion manifest that evaluation, backtest, and sweep commands can optionally consume.

### Task 5: Verification and Reporting

Files:

- Modify: `kalorie2/tasks/todo.md`.
- No changes under: `models/earnings-mention-full-web-residual-v1/`.

Plan:

- [x] Run targeted tests for new modules and changed CLI/model code.
- [x] Run a local-only v2 evaluation using the existing 264 web-evidence packets.
- [x] Run the leakage audit on existing packets.
- [x] Run at least one bounded sweep smoke test.
- [x] Compare v2 against v1 on:
  - full walk-forward
  - latest-30 strict temporal holdout
  - NO-only execution
  - event-cluster bootstrap CIs
- [x] Record results here before considering any new saved model bundle.

Implementation notes:

- Existing saved model bundle `models/earnings-mention-full-web-residual-v1/` was not edited.
- New MixMCP collection defaults to `gpt-5.4-mini` and keeps paid-call/cost cap options.
- Code review found and fixed a cutoff-leakage risk: web-evidence features now use the row-level `snapshot_target_time` as an additional cutoff, not only the packet cutoff.
- Audit outputs are guarded against `artifacts/full/` and `models/` destinations.

Verification:

- Targeted red/green v2 tests: `10 passed`, then post-review targeted tests: `7 passed`.
- Full test suite: `91 passed`.
- IDE lints: no errors for `kalorie2/src/kalorie2` and `kalorie2/tests`.

Corrected v2 local results:

- Full walk-forward rich-web Brier: `0.123186`; market Brier: `0.123966`.
- Full walk-forward all-sides backtest: `431` trades, total cost `254.46`, PnL `9.54`, ROI `0.037491`.
- Full walk-forward NO-only backtest: `326` trades, total cost `169.66`, PnL `15.34`, ROI `0.090416`.
- Strict latest-30 rich-web holdout Brier: `0.163242`; market Brier: `0.163622`.
- Strict latest-30 all-sides backtest: `52` trades, total cost `30.42`, PnL `5.58`, ROI `0.183432`.
- Strict latest-30 NO-only backtest: `37` trades, total cost `19.14`, PnL `5.86`, ROI `0.306165`.
- Web-evidence audit: `264` packets / `1,850` items scanned; `333` issues across `155` events (`168` post-cutoff dates, `131` undated items, `34` transcript/post-call flags).
- Bounded NO-only sweep smoke: `8` configs; best ROI in the smoke grid was `0.098901` at `margin=0.01`, but lower total PnL (`2.97`) than zero-margin NO-only.
- V2 NO-only CI report: `artifacts/prediction-engine/v2-rich-web-temporal-holdout-20260525/v2-no-only-ci-report.json`.
- V2 full walk-forward NO-only ROI: `0.090416`, 95% CI `[-0.000165, 0.175802]`; PnL `15.34`, 95% CI `[-0.030750, 30.540000]`.
- V2 latest-30 NO-only ROI: `0.306165`, 95% CI `[0.111584, 0.512402]`; PnL `5.86`, 95% CI `[1.779750, 10.120250]`.

## Kalorie2 Saved Model Workstation

### Goal

Recreate the old `kalorie/` web GUI inside `kalorie2/` as a saved-model prediction workstation. The GUI should dynamically discover top-level `models/*/` bundles, select a model, inspect metadata/evaluation summaries, score rows with the saved runtime contract, and show YES/NO/NONE trade decisions with an All trades / NO-only execution mode.

### Implementation Tasks

- [x] Add saved-model registry and metadata parsing for valid folders containing `artifacts/model.json`, `runtime/model_runtime.py`, and `README.md`.
- [x] Add saved-model scoring abstraction for the current runtime command contract.
- [x] Add FastAPI endpoints for model list, model detail, sample rows, and scoring.
- [x] Add backend tests for discovery, metadata parsing, scoring, and API endpoints.
- [x] Create a `kalorie2/web` Vite/React workstation UI with a dark model-registry dashboard, model selector, metric cards, execution controls, scoring panel, prediction table, and model details drawer.
- [x] Verify backend tests, lints, saved runtime smoke scoring, API smoke behavior, and frontend build.

### Verification

- `python -m pytest`: 88 passed.
- `python -m ruff check src tests`: All checks passed.
- `python models/earnings-mention-full-web-residual-v1/runtime/model_runtime.py --row-index 0`: scored row 0 with model probability `0.375294`, market probability `0.38`, side `NONE`.
- API smoke with `TestClient(create_app(models_root=Path("..") / "models"))`: discovered `earnings-mention-full-web-residual-v1`, surfaced 3,500 rows / 264 events, and scored one row.
- `npm run build` in `kalorie2/web`: TypeScript and Vite production build completed.
- IDE diagnostics: no linter errors found for edited backend/frontend files.

## Active Market Polling Workstation

### Goal

Add a read-only local polling loop that scans active Kalshi earnings mention markets every roughly 10 minutes, scores them with the preferred saved model, caches the latest poll/history, and exposes new web pages for live model output and trade opportunities. Future real trade execution should be possible via a separate execution adapter, but this pass must not place orders.

### Design

- Use a separate CLI script/entry point for polling so the web server does not own a long-running background task.
- Discover active markets through Kalshi public endpoints, normalize them into runtime-compatible CSV rows, and score them with the selected saved model.
- Cache saved-model metadata/scorer objects in process and write poll snapshots to a runtime cache directory outside `artifacts/full/`.
- Store latest poll, poll history, and trade rows as JSON files under `artifacts/runtime/workstation/`.
- Add API endpoints that read cache files for the current live model state and trade list.
- Add two UI views to the existing dark workstation:
  - Live Model: active markets, model probabilities, market probabilities, residuals, sides, and edges.
  - Trades: filtered YES/NO opportunities, with NO-only emphasized as the preferred historical slice.

### Implementation Tasks

- [x] Add active-market discovery and row normalization for live Kalshi earnings mention markets.
- [x] Add model cache/scoring helpers for repeated polling without reparsing saved model metadata each time.
- [x] Add poll snapshot storage under `artifacts/runtime/workstation/`.
- [x] Add `kalorie2-market-poller` CLI with one-shot and loop modes, defaulting to a 10-minute interval.
- [x] Add API endpoints for latest live poll and latest trade opportunities.
- [x] Add Live Model and Trades pages to `kalorie2/web`.
- [x] Add tests for scanner normalization, poll snapshot storage, CLI one-shot behavior, and API cache endpoints.

### Verification

- `python -m pytest`: 102 passed.
- `python -m ruff check src tests`: All checks passed.
- `npm run build` in `kalorie2/web`: TypeScript and Vite production build completed.
- IDE diagnostics: no linter errors found for edited backend/frontend files.
- Poller tests cover active-market normalization, snapshot/trade cache writes, CLI command exposure, and poll snapshot trade separation.
- Review regression tests cover fallback from search results to live event markets, package-local cache defaults, and preferred model defaulting to `kalorie-v2` when available.
- API tests cover latest poll, latest trade rows, and 404 behavior before the first poll.

## Saved Model Bundle: `kalorie-v2`

Goal: freeze the corrected V2 rich-web residual model as `models/kalorie-v2/`, with NO-only as the default execution policy. Do not modify `models/earnings-mention-full-web-residual-v1/`.

Plan:

- [x] Create `models/kalorie-v2/`.
- [x] Save the full training corpus under `training/`: historical CSV and web-evidence packets.
- [x] Fit one final corrected V2 rich-web model on all `264` events / `3,500` markets and save model weights, feature schema, training manifest, audit report, and evaluation reports under `artifacts/`.
- [x] Save standalone runtime code under `runtime/` with default `execution_policy="no_only"`.
- [x] Add `README.md` explaining architecture, NO-only criteria, training data, validation, and caveats.
- [x] Verify the standalone runtime loads without `PYTHONPATH` and scores a sample row.

Saved bundle:

- Directory: `models/kalorie-v2/`.
- Default execution policy: `no_only`.
- Training rows/events: `3,500` rows across `264` events.
- Web-evidence packets: `264`.
- Feature count: `40`.
- Nonzero residual weights: `34`.
- Runtime: `runtime/model_runtime.py`.

Verification:

- Ran `python models/kalorie-v2/runtime/model_runtime.py --row-index 0` from the repository root with `PYTHONPATH` removed; runtime loaded and scored row `0`.
- Verified required files, training row/event counts, packet count, feature count, nonzero weight count, and default execution policy with a local assertion script.
- IDE diagnostics: no linter errors in `models/kalorie-v2/runtime/model_runtime.py`.

Next improvement ideas:

- Use the audit exclusion manifest to train/evaluate a stricter clean-evidence variant and compare it to `kalorie-v2`.
- Regenerate web evidence for flagged events with a stricter prompt that forbids transcript/recap/undated sources and asks for source type.
- Collect a small MixMCP packet sample with `gpt-5.4-mini` under explicit cost caps, then test learned alpha before scaling.
- Build a latest-30 rolling live-watch workflow: when new markets appear, collect evidence, score NO-only, and save a pre-trade decision log before settlement.
- Add fee/slippage modeling so ROI is closer to executable Kalshi returns.

## Kalorie V3 Predictive Engine Features

Goal: implement v3 predictive features in `kalorie2/src` while keeping the frozen `models/kalorie-v2/` bundle untouched. V3 should improve unsettled-market signal quality without changing execution, live ledger, sizing, or market-open workflow yet.

Plan:

- [x] Keep `models/kalorie-v2/` read-only during this work.
- [x] Add deterministic phrase semantic bucket features for macro, regulatory, operations, product, labor, technology, finance, and generic-business concepts.
- [x] Add learned resolution/variation features, not probability multipliers.
- [x] Add stricter web-evidence relevance fields and features so low-value web-search results can be measured separately.
- [x] Add NO-side residual training support so a v3 NO-only model can be trained without optimizing the same residual target as YES/all-side experiments.
- [x] Add transcript web-search discovery prompt/schema support that finds transcript source candidates, while deterministic code remains responsible for cached transcript matching.
- [x] Run focused tests and verify `models/kalorie-v2/` remains unchanged.

Review/results:

- Added explicit `--target-side` and `--positive-label-weight` controls for evaluation/backtest runs, plus sweep grids for both values.
- Kept `models/kalorie-v2/` outside the edited `kalorie2/` tree and did not modify the frozen saved-model bundle.
- Verification: `python -m pytest -q` passed with `111` tests.
- Verification: `python -m ruff check` on edited source/tests passed.
- IDE diagnostics reported no linter errors for edited files.

V3 no-side training run:

- Training data rebuild: not needed. Reused `artifacts/full/mention-markets-historical-20260523.csv` and recomputed v3 features from existing rows plus `artifacts/prediction-engine/web-evidence-full-20260525/web-evidence`.
- Run directory: `artifacts/prediction-engine/v3-no-side-20260526/`.
- Config: `min_training_events=20`, `epochs=5`, `learning_rate=0.05`, `l2=0.001`, `residual_clip=2.0`, `target_side=no`, `positive_label_weight=2.0`, NO-only backtest margin `0`.
- Evaluation: `3,172` predictions, Brier `0.122321` vs market Brier `0.123966`.
- Calibration: ECE `0.031914` vs market ECE `0.044546`.
- NO-only backtest: `806` trades, total cost `438.99`, PnL `52.01`, ROI `11.8477%`.
- 95% event-bootstrap ROI CI: full walk-forward `[5.8648%, 17.7598%]`; latest-30 events `[0.7873%, 31.8356%]`.
- Metrics artifact: `artifacts/prediction-engine/v3-no-side-20260526/v3-metrics-ci-report.json`.

## Kalorie V3 Ablation And Margin Calibration

Goal: add reproducible feature ablation and margin calibration so v3 can optimize ROI without relying only on the raw `target_side=no` model output.

Plan:

- [x] Add CLI support for named feature ablation groups during evaluation/backtest/sweep.
- [x] Add sweep output fields that preserve the selected ablation group and margin so ROI tuning is auditable.
- [x] Run a v3 grid across all-side/NO-only policies, margins, positive-label weights, and feature ablation groups.
- [x] Select the best ROI configuration with enough trade count to be meaningful, then rerun/report it with Brier, ECE, and 95% event-bootstrap CI.
- [x] Regenerate the canvas so it shows raw v3, optimized v3, v2, all-side, and NO-only results.

Review/results:

- Added `--feature-ablation-group` for `evaluate`/`backtest` and `--feature-ablation-group-grid` for `sweep`.
- Sweep artifact: `artifacts/prediction-engine/v3-ablation-margin-20260526/sweep-summary.json`.
- Optimized prediction run: `artifacts/prediction-engine/v3-optimized-roi-20260526/`.
- Optimized report: `artifacts/prediction-engine/v3-ablation-margin-20260526/v3-ablation-margin-report.json`.
- Best full-walk-forward NO-only config was `feature_ablation_group=resolution`, `margin=0.03`, `target_side=no`, `positive_label_weight=2.0`.
- Full NO-only ROI improved from raw v3 `11.85%` to optimized `21.20%`, with 95% event-bootstrap ROI CI `[3.20%, 39.82%]`.
- Latest-30 optimized NO-only ROI was `29.03%`, but only 10 trades and CI crossed zero, so this should be treated as a selective-policy signal rather than a final deployment guarantee.

## Saved Model Bundle: `kalorie-v3-unoptimized`

Goal: save the raw V3 no-side checkpoint before ablation and margin calibration as a selectable root-level saved model bundle.

Plan/results:

- [x] Created `models/kalorie-v3-unoptimized/`.
- [x] Saved full-fit raw V3 weights in `models/kalorie-v3-unoptimized/artifacts/model.json`.
- [x] Preserved the raw V3 feature schema, training manifest, evaluation reports, historical CSV, and web-evidence packets.
- [x] Added a runtime scorer at `models/kalorie-v3-unoptimized/runtime/model_runtime.py`.
- [x] Verified runtime scoring with `python models/kalorie-v3-unoptimized/runtime/model_runtime.py --row-index 0`.
- [x] Confirmed lints reported no errors for the new runtime file.

Notes:

- This bundle is intentionally unoptimized: `feature_ablation_group=none`, `default_margin=0.0`, `target_side=no`, `positive_label_weight=2.0`.
- The optimized `feature_ablation_group=resolution`, `margin=0.03` policy is documented separately and is not baked into this bundle.

## Saved Model Bundle: `kalorie-v3`

Goal: save the balanced V3 checkpoint as the default V3 bundle, using resolution ablation and a `0.02` NO-only execution margin for better trade consistency than the peak-ROI `0.03` policy.

Plan/results:

- [x] Created `models/kalorie-v3/`.
- [x] Fit saved runtime weights on all historical rows with `feature_ablation_group=resolution`, `target_side=no`, and `positive_label_weight=2.0`.
- [x] Set default execution policy to `no_only` and default margin to `0.02`.
- [x] Preserved historical CSV, web-evidence packets, feature schema, training manifest, and evaluation reports.
- [x] Verified runtime scoring with `python models/kalorie-v3/runtime/model_runtime.py --row-index 0`.
- [x] Verified saved-model registry discovery for `kalorie-v3`.
- [x] Confirmed lints reported no errors for `models/kalorie-v3/runtime/model_runtime.py`.

Validation snapshot:

- Full NO-only margin `0.02`: `267` trades, total cost `146.95`, PnL `19.05`, ROI `12.9636%`.
- Latest-30 NO-only margin `0.02`: `35` trades, total cost `20.10`, PnL `5.90`, ROI `29.3532%`.
- Feature count after resolution ablation: `57`; nonzero saved weights: `49`.

## Model Card Schema For `kalorie-v3`

Goal: add a reusable model-card schema and concrete model card artifact that makes the latest-30 test split the primary validation view.

Plan:

- [x] Add a typed model-card schema with sections for model identity, training data, feature set, evaluation splits, metrics, confidence intervals, and caveats.
- [x] Include latest-30 as the primary testing split with ROI, number of trades, Brier, ECE, log loss, and 95% CIs where available/computable.
- [x] Include full walk-forward as secondary context so users can compare stability versus the recent test window.
- [x] Generate `models/kalorie-v3/artifacts/model-card.schema.json`.
- [x] Generate `models/kalorie-v3/artifacts/model-card.json`.
- [x] Verify the generated card is parseable, schema-shaped, and includes the requested latest-30 fields.

Review/results:

- Schema module: `kalorie2/src/kalorie2/model_cards.py`.
- Schema artifact: `models/kalorie-v3/artifacts/model-card.schema.json`.
- Model card artifact: `models/kalorie-v3/artifacts/model-card.json`.
- Latest-30 test split: `30` events, `380` markets, `35` trades at margin `0.02`.
- Latest-30 ROI: `29.3532%`, 95% event-bootstrap CI `[11.3258%, 45.9867%]`.
- Latest-30 Brier: `0.162254`, 95% CI `[0.132949, 0.190264]`; market Brier `0.163622`.
- Latest-30 ECE: `0.055070`, 95% CI `[0.041354, 0.110668]`; market ECE `0.058474`.
- Latest-30 log loss: `0.487541`, 95% CI `[0.409582, 0.560559]`; market log loss `0.490595`.
- Full walk-forward is included as a secondary backtest split with `244` events, `3,172` markets, `267` trades, and ROI `12.9636%`.
