import { RefreshCcw, Satellite } from 'lucide-react'

import { PollPredictionTable } from '@/components/PollPredictionTable'
import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

export interface LiveModelPageProps {
  readonly snapshot: PollSnapshot | null
  readonly loading: boolean
  readonly onRefresh: () => void
}

export function LiveModelPage({ snapshot, loading, onRefresh }: LiveModelPageProps) {
  const noOnlyCount = snapshot?.trade_rows.filter((row) => row.side === 'NO').length ?? null
  const selectedTrade = snapshot?.trade_rows[0] ?? null

  return (
    <section className="space-y-4">
      <div className="relative overflow-hidden rounded-2xl border border-line bg-panelStrong/70 p-5 shadow-terminal">
        <SignalLine />
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.32em] text-cyan">
              <Satellite size={14} />
              Live Model
            </p>
            <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">
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
            className="inline-flex items-center gap-2 rounded-xl border border-cyan/40 px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-cyan transition hover:bg-cyan/10"
          >
            <RefreshCcw size={16} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <PollMetric label="Poll id" value={snapshot?.poll_id ?? '--'} />
        <PollMetric label="Model" value={snapshot?.model_name ?? '--'} />
        <PollMetric label="Markets" value={formatInteger(snapshot?.market_count)} />
        <PollMetric label="NO-only" value={formatInteger(noOnlyCount)} tone="text-green" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_23rem]">
        <section className="rounded-2xl border border-line bg-panel/82 p-4 shadow-terminal">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">
                All active markets
              </p>
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

        <aside className="space-y-3">
          <SelectedTradeCard trade={selectedTrade} />
          <CriteriaCard />
          <HealthCard snapshot={snapshot} />
        </aside>
      </div>
    </section>
  )
}

interface PollMetricProps {
  readonly label: string
  readonly value: string
  readonly tone?: string
}

function PollMetric({ label, value, tone = 'text-foreground' }: PollMetricProps) {
  return (
    <div className="rounded-xl border border-line bg-panel/75 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className={`mt-2 break-all font-mono text-lg font-semibold leading-6 ${tone}`}>{value}</p>
    </div>
  )
}

function SignalLine() {
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 opacity-60">
      <svg className="h-full w-full" preserveAspectRatio="none" viewBox="0 0 800 80">
        <path
          d="M0 58 C80 48 98 28 154 38 S242 64 298 45 388 22 444 36 538 72 602 48 688 22 800 30"
          fill="none"
          stroke="rgb(54 216 255 / 0.42)"
          strokeWidth="1.5"
        />
      </svg>
    </div>
  )
}

function SelectedTradeCard({ trade }: { readonly trade: PollPredictionRow | null }) {
  return (
    <div className="rounded-2xl border border-line bg-panel/82 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-cyan">Selected trade</p>
      {trade ? (
        <>
          <p className="mt-3 break-all font-mono text-sm font-semibold text-foreground">
            {trade.market_ticker}
          </p>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <MiniStat label="Model" value={formatProbability(trade.model_probability)} />
            <MiniStat label="Market" value={formatProbability(trade.market_probability)} />
            <MiniStat label="Residual" value={formatSigned(trade.residual_delta)} tone="text-red" />
            <MiniStat label="Edge" value={formatSigned(trade.edge)} tone="text-green" />
          </div>
          <div className="mt-3 rounded-xl border border-green/25 bg-green/10 px-3 py-2 font-mono text-xs uppercase tracking-[0.16em] text-green">
            {trade.side} / execution disabled
          </div>
        </>
      ) : (
        <p className="mt-3 text-sm leading-6 text-muted">No cached trade selected yet.</p>
      )}
    </div>
  )
}

function CriteriaCard() {
  return (
    <div className="rounded-2xl border border-line bg-panel/82 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">Trade criteria</p>
      <div className="mt-3 space-y-2 font-mono text-xs text-muted">
        <p>
          <span className="text-green">YES</span> when model probability exceeds ask.
        </p>
        <p>
          <span className="text-red">NO</span> when model probability falls below bid.
        </p>
        <p>
          <span className="text-amber">SKIP</span> inside the bid/ask no-trade band.
        </p>
      </div>
    </div>
  )
}

function HealthCard({ snapshot }: { readonly snapshot: PollSnapshot | null }) {
  return (
    <div className="rounded-2xl border border-amber/25 bg-amber/10 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-amber">Runtime guardrail</p>
      <p className="mt-3 text-sm leading-6 text-muted">
        Automated order placement is disabled. The poller only writes cached predictions and trade
        candidates.
      </p>
      <p className="mt-3 font-mono text-xs text-foreground">
        Trades: {formatInteger(snapshot?.trade_count)} / Predictions:{' '}
        {formatInteger(snapshot?.prediction_count)}
      </p>
    </div>
  )
}

function MiniStat({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div className="rounded-lg border border-line/70 bg-background/60 p-2">
      <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-muted">{label}</p>
      <p className={`mt-1 font-mono text-sm font-semibold ${tone}`}>{value}</p>
    </div>
  )
}
