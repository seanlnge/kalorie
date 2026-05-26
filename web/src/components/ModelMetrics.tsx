import { BarChart3, Database, Layers3, Sigma } from 'lucide-react'
import type { ReactNode } from 'react'

import { formatCurrency, formatInteger, formatProbability } from '@/lib/format'
import type { EvaluationSnapshot, SavedModelMetadata } from '@/lib/types'

export interface ModelMetricsProps {
  readonly model: SavedModelMetadata | null
}

export function ModelMetrics({ model }: ModelMetricsProps) {
  const training = model?.training
  const snapshots = model?.evaluation_snapshots ?? []

  return (
    <section className="grid gap-4 xl:grid-cols-[1fr_1.6fr]">
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          icon={<Database size={18} />}
          label="Training rows"
          value={formatInteger(training?.row_count)}
        />
        <MetricCard
          icon={<Layers3 size={18} />}
          label="Training events"
          value={formatInteger(training?.event_count)}
        />
        <MetricCard
          icon={<BarChart3 size={18} />}
          label="Feature count"
          value={formatInteger(training?.feature_count)}
        />
        <MetricCard
          icon={<Sigma size={18} />}
          label="Nonzero weights"
          value={formatInteger(training?.nonzero_weight_count)}
        />
      </div>

      <div className="rounded-2xl border border-line bg-panel/80 p-4 shadow-terminal">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">
              Evaluation Snapshot
            </p>
            <h2 className="font-display text-lg font-semibold">Backtest and holdout performance</h2>
          </div>
          <span className="rounded-full border border-cyan/30 bg-cyan/10 px-3 py-1 font-mono text-xs text-cyan">
            {snapshots.length || 0} reports
          </span>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {snapshots.length === 0 ? (
            <div className="rounded-xl border border-amber/30 bg-amber/10 p-4 text-sm text-amber">
              No evaluation artifact was exposed for this model.
            </div>
          ) : (
            snapshots.slice(0, 4).map((snapshot) => (
              <EvaluationCard key={snapshot.label} snapshot={snapshot} />
            ))
          )}
        </div>
      </div>
    </section>
  )
}

interface MetricCardProps {
  readonly icon: ReactNode
  readonly label: string
  readonly value: string
}

function MetricCard({ icon, label, value }: MetricCardProps) {
  return (
    <div className="rounded-2xl border border-line bg-panel/80 p-4 shadow-terminal">
      <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg border border-cyan/30 bg-cyan/10 text-cyan">
        {icon}
      </div>
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className="mt-2 font-mono text-2xl font-semibold tracking-tight text-foreground">{value}</p>
    </div>
  )
}

interface EvaluationCardProps {
  readonly snapshot: EvaluationSnapshot
}

function EvaluationCard({ snapshot }: EvaluationCardProps) {
  const positive = (snapshot.pnl ?? 0) >= 0 && (snapshot.roi ?? 0) >= 0
  return (
    <div className="rounded-xl border border-line/70 bg-background/55 p-3">
      <p className="font-mono text-sm font-semibold text-foreground">{snapshot.label}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <SmallMetric label="Trades" value={formatInteger(snapshot.trades)} />
        <SmallMetric
          label="PnL"
          value={formatCurrency(snapshot.pnl)}
          tone={positive ? 'text-green' : 'text-red'}
        />
        <SmallMetric
          label="ROI"
          value={formatProbability(snapshot.roi)}
          tone={positive ? 'text-green' : 'text-red'}
        />
      </div>
      {snapshot.brier !== null && snapshot.brier !== undefined ? (
        <p className="mt-3 font-mono text-xs text-muted">
          Brier {snapshot.brier.toFixed(6)}
          {snapshot.market_brier ? ` vs market ${snapshot.market_brier.toFixed(6)}` : ''}
        </p>
      ) : null}
    </div>
  )
}

interface SmallMetricProps {
  readonly label: string
  readonly value: string
  readonly tone?: string
}

function SmallMetric({ label, value, tone = 'text-foreground' }: SmallMetricProps) {
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-wide text-muted">{label}</p>
      <p className={`mt-1 font-mono font-semibold ${tone}`}>{value}</p>
    </div>
  )
}
