export type ExecutionMode = 'all' | 'no_only'

export type RiskTradeSide = 'all' | 'no_only' | 'yes_only'

export interface RiskPreset {
  id: string
  label: string
  description: string
  trade_side: RiskTradeSide
  min_margin: number
  kelly_fraction: number
  max_position_fraction: number
  max_event_exposure_fraction: number
}

export interface TrainingSummary {
  row_count?: number | null
  event_count?: number | null
  feature_count?: number | null
  nonzero_weight_count?: number | null
  web_evidence_packet_count?: number | null
  first_event?: string | null
  last_event?: string | null
}

export interface EvaluationSnapshot {
  label: string
  trades?: number | null
  pnl?: number | null
  roi?: number | null
  brier?: number | null
  market_brier?: number | null
  notes?: string | null
}

export interface ModelCardMetricValue {
  value: number
  unit?: string | null
  description?: string | null
  ci95?: {
    low: number
    high: number
    method?: string | null
    confidence_level?: number | null
  } | null
}

export interface ModelCardEvaluationSplit {
  name: string
  role: string
  event_count: number
  market_count: number
  metrics: Record<string, ModelCardMetricValue>
  notes?: string | null
}

export interface ReturnPercentileBand {
  p10: number
  p25: number
  expected: number
  p75: number
  p90: number
}

export interface RiskPresetTrial {
  risk_preset_id: string
  label: string
  trade_side: RiskTradeSide
  min_margin: number
  kelly_fraction: number
  max_position_fraction: number
  max_event_exposure_fraction: number
  risk_of_ruin_estimate: number
  risk_of_ruin_label: string
  trade_count: number
  market_count: number
  trade_percent: number
  ev_per_10_markets: number
  expected_return_per_market: ReturnPercentileBand
}

export interface ModelCardPreview {
  split_name: string
  role?: string | null
  market_count?: number | null
  brier?: number | null
  market_brier?: number | null
  ece?: number | null
  market_ece?: number | null
  log_loss?: number | null
  market_log_loss?: number | null
}

export interface ModelCard {
  schema_version?: string
  model_name: string
  model_version?: number | null
  model_type: string
  training_data: Record<string, number | string | null>
  feature_set: Record<string, number | string | string[] | null>
  evaluation_splits: ModelCardEvaluationSplit[]
  caveats?: string[]
  recommended_use?: string | null
}

export interface SavedModelMetadata {
  name: string
  path: string
  health: 'ready'
  model_type?: string | null
  model_version?: number | null
  trained_at?: string | null
  readme: string
  readme_summary: string
  training: TrainingSummary
  evaluation_snapshots: EvaluationSnapshot[]
  artifact_paths: Record<string, string>
  model_card?: ModelCard | null
  model_card_preview?: ModelCardPreview | null
  risk_preset_trials: RiskPresetTrial[]
}

export interface SampleRow {
  row_index: string
  market_ticker?: string
  event_ticker?: string
  word_said?: string
  normalized_word_said?: string
  preclose_yes_bid?: string
  preclose_yes_ask?: string
  preclose_yes_mid?: string
}

export interface ScoreRow {
  market_ticker: string
  event_ticker: string
  model_probability: number
  market_probability: number
  residual_delta: number
  trade_decision: Record<string, unknown>
  side: 'YES' | 'NO' | 'NONE' | string
  edge: number
  cost: number
  raw: Record<string, unknown>
}

export interface ScoreResponse {
  model_name: string
  execution_mode: ExecutionMode
  rows: ScoreRow[]
}

export interface PollPredictionRow {
  market_ticker: string
  event_ticker: string
  event_datetime?: string | null
  event_title?: string | null
  target_phrase: string
  model_name: string
  risk_preset_id?: string | null
  model_probability: number
  market_probability: number
  yes_bid: number
  yes_ask: number
  residual_delta: number
  side: 'YES' | 'NO' | 'NONE' | string
  edge: number
  cost: number
  ev_per_contract?: number | null
  kelly_fraction_raw?: number | null
  recommended_fraction?: number | null
  passes_risk_filter?: boolean | null
  volume: number
  recommended_dollars?: number | null
  recommended_contracts?: number | null
}

export interface PollSnapshot {
  poll_id: string
  model_name: string
  risk_preset_id?: string | null
  started_at: string
  completed_at: string
  market_polled_at?: string | null
  model_run_started_at?: string | null
  model_run_completed_at?: string | null
  next_market_poll_at?: string | null
  next_model_run_at?: string | null
  market_count: number
  prediction_count: number
  trade_count: number
  prediction_rows: PollPredictionRow[]
  trade_rows: PollPredictionRow[]
}

export interface AccountSummary {
  available: boolean
  source: 'kalshi' | 'paper' | string
  portfolio_value?: number | null
  free_cash?: number | null
  position_exposure?: number | null
  bankroll: number
  error?: string | null
}

export interface OpenPosition {
  market_ticker: string
  side: 'YES' | 'NO' | string
  contracts: number
  average_price?: number | null
  market_value?: number | null
  exposure?: number | null
  realized_pnl?: number | null
  fees_paid?: number | null
}

export interface OpenPositionsSummary {
  available: boolean
  source: 'kalshi' | 'paper' | string
  open_position_count: number
  total_contracts: number
  average_price?: number | null
  total_market_value?: number | null
  total_exposure?: number | null
  realized_pnl?: number | null
  fees_paid?: number | null
  positions: OpenPosition[]
  error?: string | null
}
