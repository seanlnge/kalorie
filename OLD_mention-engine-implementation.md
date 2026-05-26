# Mention Engine Implementation

## 1) Goal and Optimization Target

The goal is unchanged: estimate mention probabilities for earnings mention markets with the lowest possible out-of-sample Brier score, then convert those probabilities into safer, higher-quality market edges.

Primary optimization target:

- Minimize holdout Brier on realistic company-quarter splits.

Secondary targets:

- Maintain reasonable calibration (avoid overconfident tails).
- Maintain operational speed for iterative retraining.
- Improve company-specific performance over global baseline.

## 2) What This System Predicts

Each prediction is a binary event:

- `1`: target phrase appears in the transcript/event text under the training label definition.
- `0`: target phrase does not appear.

Training is phrase-presence oriented, then adapted to market phrasing. This means:

- We train on transcript/event evidence, not direct market outcomes.
- We compare to market prices downstream.
- We emphasize phrase catalogs that match real contract wording.

## 3) Implementation Scope in This Repo

The implemented stack is centered around:

1. Evidence ingestion (SEC/transcripts/news).
2. Synthetic phrase-example generation.
3. Global model training and company retraining.
4. Evaluation with holdout Brier and diagnostics.
5. Kalshi market discovery + probability comparison.

This repo intentionally prioritizes model quality and edge discovery over observability/platform architecture.

## 4) System Components (Current)

### 4.1 Data Ingestion

Implemented collectors and clients:

- SEC materials and EX-99-style filing ingestion.
- Transcript corpus scanning from local structured files.
- NewsData ingestion with archive/latest fallback handling.
- Tiingo ingestion with typed vendor errors and yfinance fallback.
- DefeatBeta ingestion through Python-library-backed stock-news parquet workflow.

Related modules include:

- `src/kalorie/clients/newsdata.py`
- `src/kalorie/clients/tiingo.py`
- `src/kalorie/clients/defeatbeta.py`
- `src/kalorie/app/cli.py`

### 4.2 Evidence Normalization

Evidence is mapped into `PublicDocumentManifest` rows with:

- `company_symbol`
- `fiscal_year`
- `fiscal_quarter`
- `source_type`
- `published_at`
- `fetched_at`
- `raw_path`
- `content_hash`

This manifest abstraction lets us merge SEC + transcript-adjacent docs + news into one training input surface.

### 4.3 Feature Building

Dataset builder creates `HistoricalTrainingExample` rows from:

- transcript records for each company-quarter,
- evidence docs filtered by event timing,
- target phrase list (default + market expansions + company-specific terms),
- feature extraction routines over phrase/evidence context.

Important behavior:

- global phrase set can be expanded from historical market contracts,
- company-specific targets can be injected,
- evidence reliability features are included in feature vectors.

### 4.4 Modeling

Primary model path:

- `model1` global training + optional optimization.
- company-level retraining via `train-model1-company`.

Company retraining knobs include:

- `--regularization-c`
- `--class-weight-balanced`
- `--include-target-indicator`
- `--recency-ema-half-life-quarters`

### 4.5 Evaluation

Main reported metric:

- Holdout Brier on time-aware event partitions.

Support metrics:

- in-sample Brier,
- log-loss in selected evaluation passes,
- ablation summaries across data source variants.

## 5) Phrase Strategy (Critical)

The biggest practical shift from the original generic business setup is phrase realism:

- Do not rely only on generic terms like `revenue`, `margin`, `pricing`.
- Include Kalshi-style and company-idiosyncratic wording.
- Use market-derived phrase expansion where possible.

Examples that matter in this project:

- `omnichannel`
- `openai`
- `salmon`
- `sweet potato`
- `auv`

This has shown measurable benefit in the existing Walmart experiments compared with generic-only phrasing.

## 6) News Feature Strategy (Current State)

News is integrated as a feature family, not as a standalone model.

Implemented ideas:

- opinion vs relevant flavor tagging,
- source reliability encoding in `source_type`,
- aggregate reliability features in dataset construction,
- optional pre-earnings window harvesting per quarter,
- EMA recency weighting in company retraining.

Important caveat:

- News can hurt holdout Brier if coverage is noisy, sparse, or mismatched to quarters.
- News should remain ablation-driven, not blindly default-on.

## 7) DefeatBeta Implementation (Current)

DefeatBeta integration is now aligned to the Python library/data modality:

- Reads stock news from DefeatBeta-backed parquet source.
- Filters by `related_symbols` and date range.
- Parses article body from `news` paragraph payloads.
- Maps output into repo-native manifest/evidence format.

CLI command:

- `collect-defeatbeta-pre-earnings-week-articles`

Current runtime design:

- One full-range pull for the symbol.
- Local slicing into per-quarter pre-earnings windows.
- Optional yfinance fallback for empty windows/errors.

This replaced earlier direct-host assumptions and significantly reduced command runtime for Walmart runs.

## 8) Kalshi Integration (Current)

Kalshi support is focused on market discovery and probability comparison:

- Company/event market discovery helpers.
- Fallback logic for endpoint differences and pagination.
- Price parsing aligned to dollar-denominated fields when available.
- Side-by-side model vs market probability comparisons.

This is operationally important because contract wording often diverges from generic training phrase lists.

## 9) Core CLI Workflows

### 9.1 Data Collection

- `collect-sec-press-releases-for-transcripts`
- `collect-newsdata-company-articles`
- `collect-tiingo-company-articles`
- `collect-defeatbeta-pre-earnings-week-articles`
- `discover-company-earnings-markets`

### 9.2 Dataset Build

- `build-synthetic-phrase-dataset`

Inputs typically include:

- transcript root,
- merged manifests,
- optional market contracts file for phrase expansion,
- template phrase catalog.

### 9.3 Training

- `train-model1-optimized` for baseline/global optimization passes.
- `train-model1-company` for company adaptation.

### 9.4 Evaluation and Ablation

- `run-base-ablation-harness`
- model-specific Brier/log-loss evaluation scripts and outputs in `artifacts/model1/eval`.

## 10) Data and Artifact Conventions

Key directories used in current workflow:

- `data/earnings_call_transcripts/` for processed transcript files.
- `data/sec/manifests/` and `data/manifests/` for normalized evidence manifests.
- `data/raw/` for collected raw text artifacts by source.
- `artifacts/model1/datasets/` for built synthetic datasets.
- `artifacts/model1/models/` for trained model artifacts.
- `artifacts/model1/eval/` for evaluation reports.

This layout is intentional: evidence capture and model outputs stay reproducible and inspectable.

## 11) Evaluation Principles

Rules this implementation follows:

- Prefer time-aware splits over random-row splits.
- Report holdout Brier as primary metric.
- Compare variants with controlled ablations, not one-off impressions.
- Separate in-sample from holdout conclusions.
- Treat any news uplift claim as invalid unless coverage diagnostics confirm meaningful historical presence.

## 12) Known Failure Modes

1. Phrase mismatch:

- Training catalog differs from actual market wording.
- Effect: weak live relevance, poor contract alignment.

1. News coverage illusion:

- News pipeline runs, but quarter coverage is shallow or recent-only.
- Effect: "news added" with no real informational gain.

1. Overfitting to sparse company windows:

- Excess confidence from small or uneven quarter counts.
- Effect: in-sample gain, holdout regression.

1. Timing leakage:

- Evidence published after the intended decision point.
- Effect: inflated backtest quality.

1. Unseen-target instability:

- Rare/new phrases with weak historical basis.
- Effect: overly aggressive probabilities and unsafe edges.

## 13) Current Practical Status (As Of Latest Updates)

- Phrase expansion and Kalshi wording alignment are implemented and high value.
- Company market discovery helper path is implemented and tested.
- Multi-provider news ingestion is implemented end-to-end.
- DefeatBeta pre-earnings-week workflow is implemented through the correct library-backed modality.
- News uplift remains conditional; some runs regress holdout Brier without stricter quality gating.
- Event-baseline evidence semantics are now class-aware: SEC and earnings press-release evidence can be retained as event-matched baseline material even when local timestamps fall after the T-10 cutoff, while news and other time-sensitive sources still obey `call_start - 10 minutes`.
- Synthetic phrase-market training is first-class. Most historical training rows may be artificial "would Kalshi have listed this word/phrase?" rows, settled by eligible transcript containment under Kalshi earnings-mention wording rules.
- Event scenario catalogs and scenario embedding features are implemented as optional event-level inputs. Scenario generation uses only pre-call materials and target phrases; transcripts remain label-only.
- Real Kalshi rows are reserved for benchmark/effect checks. The benchmark report compares model probability against Kalshi yes-bid, yes-ask, and yes-mid with Brier/ECE deltas.
- Latest local smoke artifacts:
  - Dataset: `artifacts/model1/datasets/smoke-synthetic-nvda-evidence-candidates-20260521.json` (`520` rows, `13` events, gate-clean).
  - Gate report: `artifacts/model1/eval/smoke-model1-nvda-evidence-candidates-gate-20260521.json`.
  - Model: `artifacts/model1/models/smoke-model1-nvda-evidence-candidates-20260521.json`.
  - Kalshi benchmark JSON: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.json`, where the existing base model scored Brier `0.111456` versus Kalshi mid Brier `0.197245` on the cached 5-event pack.
  - Kalshi benchmark table: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.table.md`.
  - Kalshi benchmark detail log: `artifacts/model1/eval/model1-base-cutoff2h-vs-kalshi-preclose-20260521.log`.
- Caveat: the cached event-pack quote snapshots are pre-close snapshots, not yet guaranteed exact `call_start - 10 minutes` snapshots. Future benchmark rows should prefer exact T-10 quotes and record nearest-snapshot deltas when exact quotes are unavailable.

## 14) Immediate Next Work (High Priority)

1. Pre-train quality gate:

- Add hard diagnostics for news coverage and quality.
- Optionally abort training if thresholds are not met.

1. Unseen-target confidence controls:

- Penalize or shrink probabilities for weakly supported rare phrases.

1. Structured news ablations:

- opinion-only,
- high-trust publisher-only,
- preweek-only vs broader windows,
- and source-by-source incremental tests.

1. Continuous market wording updates:

- Keep target phrases synchronized with current contract styles and emerging terms.

## 15) Implementation Philosophy

This mention engine should remain:

- metric-first,
- evidence-aware,
- ablation-driven,
- and ruthless about calibration and data quality.

The end objective is not architectural elegance. The end objective is persistent probability edge on real mention markets with defensible, reproducible model behavior.
