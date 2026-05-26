export interface MarketRow {
  market_ticker: string
  event_ticker: string
  company_symbol: string
  title: string
  target_phrase: string
  yes_bid: string
  yes_ask: string
  spread: string
  volume: number
}

export interface MarketEventRow {
  event_ticker: string
  company_symbol: string
  market_count: number
  total_volume: number
  representative_market_ticker: string
  representative_phrase: string
}

export interface RunInfo {
  run_id: string
  run_dir: string
  market_ticker: string
  created_at: string
  status: string
}

export interface RunResultPayload {
  run_id: string
  market_ticker: string
  rows: Array<Record<string, unknown>>
}

export interface JobInfo {
  job_id: string
  idempotency_key: string
  market_ticker: string
  status: string
  wait_reason?: string | null
  created_at: string
}

