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
