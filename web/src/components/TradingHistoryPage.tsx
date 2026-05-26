import { History } from 'lucide-react'

import { PollPredictionTable } from '@/components/PollPredictionTable'
import { formatInteger } from '@/lib/format'
import type { PollSnapshot } from '@/lib/types'

export interface TradingHistoryPageProps {
  readonly history: readonly PollSnapshot[]
}

export function TradingHistoryPage({ history }: TradingHistoryPageProps) {
  const totalTrades = history.reduce((total, snapshot) => total + snapshot.trade_count, 0)
  const latest = history[0] ?? null

  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-line bg-panelStrong/80 p-5 shadow-terminal">
        <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.28em] text-green">
          <History size={14} />
          Trading History
        </p>
        <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight">
          Bot decision history
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
          Cached trade candidates from each poll. This is intentionally read-only decision history
          until real Kalshi execution/fill records are wired in.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <HistoryMetric label="Polls" value={formatInteger(history.length)} />
        <HistoryMetric label="Trade candidates" value={formatInteger(totalTrades)} tone="text-green" />
        <HistoryMetric label="Latest poll" value={latest?.poll_id ?? '--'} />
        <HistoryMetric
          label="Latest completed"
          value={latest ? new Date(latest.completed_at).toLocaleString() : '--'}
        />
      </div>

      <div className="space-y-3">
        {history.length === 0 ? (
          <div className="rounded-lg border border-amber/35 bg-amber/10 p-5 text-sm text-amber">
            No poll history found. Run `kalorie2-market-poller once` or `loop` to seed this page.
          </div>
        ) : (
          history.map((snapshot) => (
            <section
              key={snapshot.poll_id}
              className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal"
            >
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
                    Poll {snapshot.poll_id}
                  </p>
                  <h2 className="font-display text-lg font-semibold">
                    {snapshot.trade_count} trade candidates from {snapshot.prediction_count} predictions
                  </h2>
                </div>
                <p className="font-mono text-xs text-muted">
                  {new Date(snapshot.completed_at).toLocaleString()}
                </p>
              </div>
              <PollPredictionTable
                rows={snapshot.trade_rows}
                emptyMessage="This poll produced no trade candidates."
              />
            </section>
          ))
        )}
      </div>
    </section>
  )
}

function HistoryMetric({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div className="rounded-lg border border-line bg-panel/75 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className={`mt-2 break-all font-mono text-lg font-semibold leading-6 ${tone}`}>{value}</p>
    </div>
  )
}
