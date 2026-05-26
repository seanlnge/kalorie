export type ExecutionMode = 'all' | 'no_only'

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
  policy: string
  margin: number
  metrics: Record<string, ModelCardMetricValue>
  notes?: string | null
}

export interface ModelCardPreview {
  split_name: string
  role?: string | null
  policy?: string | null
  trade_count?: number | null
  market_count?: number | null
  trade_percent?: number | null
  brier?: number | null
  roi_on_cost?: number | null
  ev_per_10_trades?: number | null
}

export interface ModelCard {
  schema_version?: string
  model_name: string
  model_version?: number | null
  model_type: string
  default_execution_policy: string
  default_margin: number
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
  target_phrase: string
  model_name: string
  model_probability: number
  market_probability: number
  yes_bid: number
  yes_ask: number
  residual_delta: number
  side: 'YES' | 'NO' | 'NONE' | string
  edge: number
  cost: number
  volume: number
}

export interface PollSnapshot {
  poll_id: string
  model_name: string
  started_at: string
  completed_at: string
  market_count: number
  prediction_count: number
  trade_count: number
  prediction_rows: PollPredictionRow[]
  trade_rows: PollPredictionRow[]
}
