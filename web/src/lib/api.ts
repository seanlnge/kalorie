import type {
  AccountSummary,
  ExecutionMode,
  OpenPositionsSummary,
  PollPredictionRow,
  PollSnapshot,
  RiskPreset,
  RiskPresetTrial,
  SampleRow,
  SavedModelMetadata,
  ScoreResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export async function listModels(): Promise<SavedModelMetadata[]> {
  const payload = await request<{ models: SavedModelMetadata[] }>('/api/models')
  return payload.models
}

export async function listRiskPresets(): Promise<RiskPreset[]> {
  const payload = await request<{ risk_presets: RiskPreset[] }>('/api/risk-presets')
  return payload.risk_presets
}

export async function getAccountSummary(): Promise<AccountSummary> {
  const payload = await request<{ summary: AccountSummary }>('/api/account/summary')
  return payload.summary
}

export async function getOpenPositionsSummary(): Promise<OpenPositionsSummary> {
  const payload = await request<{ summary: OpenPositionsSummary }>('/api/account/positions')
  return payload.summary
}

export async function getModel(modelName: string): Promise<SavedModelMetadata> {
  const payload = await request<{ model: SavedModelMetadata }>(
    `/api/models/${encodeURIComponent(modelName)}`,
  )
  return payload.model
}

export async function getSampleRows(modelName: string): Promise<SampleRow[]> {
  const payload = await request<{ rows: SampleRow[] }>(
    `/api/models/${encodeURIComponent(modelName)}/sample-rows`,
  )
  return payload.rows
}

export async function scoreModel({
  modelName,
  executionMode,
  rowIndex,
  csvFile,
}: {
  modelName: string
  executionMode: ExecutionMode
  rowIndex: number
  csvFile?: File | null
}): Promise<ScoreResponse> {
  const formData = new FormData()
  formData.append('execution_mode', executionMode)
  formData.append('row_index', String(rowIndex))
  if (csvFile) {
    formData.append('csv_file', csvFile)
  }
  return request<ScoreResponse>(`/api/models/${encodeURIComponent(modelName)}/score`, {
    method: 'POST',
    body: formData,
  })
}

export async function computeRiskTrial(
  modelName: string,
  riskPreset: RiskPreset,
): Promise<RiskPresetTrial> {
  const payload = await request<{ trial: RiskPresetTrial }>(
    `/api/models/${encodeURIComponent(modelName)}/risk-trial`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ risk_preset: riskPreset }),
    },
  )
  return payload.trial
}

export async function getLatestPoll(): Promise<PollSnapshot | null> {
  try {
    const payload = await request<{ snapshot: PollSnapshot }>('/api/polls/latest')
    return payload.snapshot
  } catch {
    return null
  }
}

export async function getLatestTrades(): Promise<PollPredictionRow[]> {
  const payload = await request<{ trades: PollPredictionRow[] }>('/api/trades/latest')
  return payload.trades
}

export async function getCurrentMarkets(
  modelName: string,
  riskPreset: RiskPreset,
  options: { readonly refreshMarkets?: boolean } = {},
): Promise<PollSnapshot> {
  const params = new URLSearchParams({
    risk_preset_id: riskPreset.id,
    refresh_markets: String(options.refreshMarkets ?? true),
  })
  const payload = await request<{ snapshot: PollSnapshot }>(
    `/api/models/${encodeURIComponent(modelName)}/current-markets?${params.toString()}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ risk_preset: riskPreset }),
    },
  )
  return payload.snapshot
}

export async function getPollHistory(limit = 50): Promise<PollSnapshot[]> {
  const payload = await request<{ snapshots: PollSnapshot[] }>(`/api/polls/history?limit=${limit}`)
  return payload.snapshots
}
