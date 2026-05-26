import { ShieldCheck } from 'lucide-react'

import { PollPredictionTable } from '@/components/PollPredictionTable'
import { formatInteger } from '@/lib/format'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

export interface TradesPageProps {
  readonly snapshot: PollSnapshot | null
  readonly trades: readonly PollPredictionRow[]
}

export function TradesPage({ snapshot, trades }: TradesPageProps) {
  const noTrades = trades.filter((trade) => trade.side === 'NO')
  const yesTrades = trades.filter((trade) => trade.side === 'YES')

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-line bg-panelStrong/70 p-5 shadow-terminal">
        <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.32em] text-green">
          <ShieldCheck size={14} />
          Trade Opportunities
        </p>
        <h2 className="mt-2 font-display text-3xl font-bold tracking-tight">
          Read-only trade slate
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
          These rows use the same rule as the backtests: YES above ask, NO below bid. Execution is
          intentionally disabled in this version; a future Kalshi execution adapter can consume this
          cache after explicit safeguards are added.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <TradeMetric label="All opportunities" value={formatInteger(trades.length)} />
        <TradeMetric label="NO-only slice" value={formatInteger(noTrades.length)} tone="text-green" />
        <TradeMetric label="YES slice" value={formatInteger(yesTrades.length)} />
        <TradeMetric label="Source poll" value={snapshot?.poll_id ?? '--'} />
      </div>

      <section className="rounded-2xl border border-green/30 bg-green/5 p-4 shadow-terminal">
        <div className="mb-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-green">
            Preferred historical slice
          </p>
          <h3 className="font-display text-lg font-semibold">NO-only opportunities</h3>
        </div>
        <PollPredictionTable
          rows={noTrades}
          emptyMessage="No NO opportunities in the latest cached poll."
        />
      </section>

      <section className="rounded-2xl border border-line bg-panel/80 p-4 shadow-terminal">
        <div className="mb-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">All trades</p>
          <h3 className="font-display text-lg font-semibold">YES and NO opportunities</h3>
        </div>
        <PollPredictionTable rows={trades} emptyMessage="No opportunities in the latest cached poll." />
      </section>
    </section>
  )
}

interface TradeMetricProps {
  readonly label: string
  readonly value: string
  readonly tone?: string
}

function TradeMetric({ label, value, tone = 'text-foreground' }: TradeMetricProps) {
  return (
    <div className="rounded-xl border border-line bg-panel/75 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className={`mt-2 break-all font-mono text-lg font-semibold leading-6 ${tone}`}>{value}</p>
    </div>
  )
}
