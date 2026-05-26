import type { JobInfo, MarketEventRow, MarketRow, RunInfo, RunResultPayload } from '@/lib/types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export function websocketUrl(path: string): string {
  const url = new URL(API_BASE)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = path
  return url.toString()
}

export interface OpenMarketsPayload {
  markets: MarketRow[]
  events: MarketEventRow[]
}

export async function getOpenMarketsPayload(): Promise<OpenMarketsPayload> {
  const payload = await request<{ markets: MarketRow[]; events?: MarketEventRow[] }>('/api/markets/open')
  return {
    markets: payload.markets ?? [],
    events: payload.events ?? [],
  }
}

export async function getOpenMarkets(): Promise<MarketRow[]> {
  const payload = await getOpenMarketsPayload()
  return payload.markets
}

export async function getEventMarkets(eventTicker: string): Promise<MarketRow[]> {
  const payload = await request<{ markets: MarketRow[] }>(
    `/api/events/${encodeURIComponent(eventTicker)}/markets`
  )
  return payload.markets ?? []
}

export async function listRuns(marketTicker: string): Promise<RunInfo[]> {
  const payload = await request<{ runs: RunInfo[] }>(`/api/markets/${marketTicker}/runs`)
  return payload.runs
}

export async function getLatestRun(marketTicker: string): Promise<RunInfo | null> {
  try {
    const payload = await request<{ run: RunInfo }>(`/api/markets/${marketTicker}/runs/latest`)
    return payload.run
  } catch {
    return null
  }
}

export async function getRun(
  marketTicker: string,
  runId: string
): Promise<{ run: RunInfo; result: RunResultPayload | null }> {
  return request<{ run: RunInfo; result: RunResultPayload | null }>(
    `/api/markets/${marketTicker}/runs/${runId}`
  )
}

export async function submitJob(
  marketTicker: string,
  {
    files,
    historyWindow = 'all_available',
    dataMode = 'mixed_best_effort',
    decisionCutoffTs,
  }: {
    files: File[]
    historyWindow?: string
    dataMode?: string
    decisionCutoffTs?: string
  }
): Promise<{ job: JobInfo; run: RunInfo }> {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  formData.append('history_window', historyWindow)
  formData.append('data_mode', dataMode)
  if (decisionCutoffTs) {
    formData.append('decision_cutoff_ts', decisionCutoffTs)
  }
  const idempotencyKey = crypto.randomUUID()
  return request<{ job: JobInfo; run: RunInfo }>(`/api/markets/${marketTicker}/jobs`, {
    method: 'POST',
    body: formData,
    headers: {
      'Idempotency-Key': idempotencyKey,
    },
  })
}

export async function listJobs(): Promise<JobInfo[]> {
  const payload = await request<{ jobs: JobInfo[] }>('/api/jobs')
  return payload.jobs
}

