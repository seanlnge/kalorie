import { RefreshCcw, Satellite } from 'lucide-react'

import { PollPredictionTable } from '@/components/PollPredictionTable'
import { formatInteger } from '@/lib/format'
import type { PollSnapshot } from '@/lib/types'

export interface LiveModelPageProps {
  readonly snapshot: PollSnapshot | null
  readonly loading: boolean
  readonly onRefresh: () => void
}

export function LiveModelPage({ snapshot, loading, onRefresh }: LiveModelPageProps) {
  return (
    <section className="space-y-5">
      <div className="rounded-[2rem] border border-line/70 bg-panelStrong/50 p-6 shadow-terminal">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 text-xs uppercase tracking-[0.32em] text-cyan">
              <Satellite size={14} />
              Live Model
            </p>
            <h2 className="mt-2 font-display text-3xl font-semibold tracking-tight">
              Active-market model scan
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
              Latest read-only poll over active markets. Start `kalorie2-market-poller loop` to
              refresh this cache roughly every 10 minutes.
            </p>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-2 rounded-2xl border border-cyan/40 px-4 py-3 font-semibold text-cyan transition hover:bg-cyan/10"
          >
            <RefreshCcw size={16} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <PollMetric label="Poll id" value={snapshot?.poll_id ?? '--'} />
        <PollMetric label="Model" value={snapshot?.model_name ?? '--'} />
        <PollMetric label="Markets" value={formatInteger(snapshot?.market_count)} />
        <PollMetric label="Trades" value={formatInteger(snapshot?.trade_count)} />
      </div>

      <section className="rounded-3xl border border-line/70 bg-panel/70 p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-muted">All active markets</p>
            <h3 className="font-display text-lg font-semibold">
              {loading ? 'Loading cached poll...' : 'Model probabilities'}
            </h3>
          </div>
          <p className="font-mono text-xs text-muted">
            Completed {snapshot ? new Date(snapshot.completed_at).toLocaleString() : '--'}
          </p>
        </div>
        <PollPredictionTable
          rows={snapshot?.prediction_rows ?? []}
          emptyMessage="No poll cache yet. Run the market poller once or in loop mode."
        />
      </section>
    </section>
  )
}

interface PollMetricProps {
  readonly label: string
  readonly value: string
}

function PollMetric({ label, value }: PollMetricProps) {
  return (
    <div className="rounded-3xl border border-line/70 bg-panel/70 p-5">
      <p className="text-xs uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className="mt-2 truncate font-mono text-xl font-semibold text-foreground">{value}</p>
    </div>
  )
}
