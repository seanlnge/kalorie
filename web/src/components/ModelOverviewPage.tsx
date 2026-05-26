import { Boxes, FileJson2 } from 'lucide-react'

import { ModelMetrics } from '@/components/ModelMetrics'
import { PredictionTable } from '@/components/PredictionTable'
import { PollPredictionTable } from '@/components/PollPredictionTable'
import { ScoringPanel } from '@/components/ScoringPanel'
import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow, SampleRow, SavedModelMetadata, ScoreRow } from '@/lib/types'

export interface ModelOverviewPageProps {
  readonly model: SavedModelMetadata | null
  readonly sampleRows: readonly SampleRow[]
  readonly selectedRowIndex: number
  readonly scoring: boolean
  readonly predictions: readonly ScoreRow[]
  readonly currentMarketRows: readonly PollPredictionRow[]
  readonly onRowIndexChange: (rowIndex: number) => void
  readonly onScoreSample: () => Promise<void>
  readonly onScoreUpload: (file: File) => Promise<void>
}

export function ModelOverviewPage({
  model,
  sampleRows,
  selectedRowIndex,
  scoring,
  predictions,
  currentMarketRows,
  onRowIndexChange,
  onScoreSample,
  onScoreUpload,
}: ModelOverviewPageProps) {
  const card = model?.model_card

  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-line bg-panelStrong/80 p-5 shadow-terminal">
        <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.28em] text-cyan">
          <Boxes size={14} />
          Model Overview
        </p>
        <h1 className="mt-2 break-all font-display text-3xl font-semibold tracking-tight">
          {model?.name ?? 'No model selected'}
        </h1>
        <p className="mt-3 max-w-4xl text-sm leading-6 text-muted">
          {card?.recommended_use ?? model?.readme_summary ?? 'Select a saved model to inspect its card.'}
        </p>
      </div>

      {card ? <ModelCardPanel model={model} /> : <LegacyMetadataNotice model={model} />}

      <ModelMetrics model={model} />

      <ScoringPanel
        sampleRows={sampleRows}
        selectedRowIndex={selectedRowIndex}
        scoring={scoring}
        onRowIndexChange={onRowIndexChange}
        onScoreSample={onScoreSample}
        onScoreUpload={onScoreUpload}
      />
      <PredictionTable rows={predictions} />
      <section className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
        <div className="mb-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Current Kalshi Market Tests
          </p>
          <h2 className="font-display text-lg font-semibold">
            Latest cached active-market scores for this model
          </h2>
        </div>
        <PollPredictionTable
          rows={currentMarketRows}
          emptyMessage="No cached active-market scores for the selected model yet."
        />
      </section>
    </section>
  )
}

function ModelCardPanel({ model }: { readonly model: SavedModelMetadata | null }) {
  const card = model?.model_card
  if (!card) return null

  return (
    <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
      <div className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
        <div className="mb-4 flex items-center gap-2">
          <FileJson2 size={16} className="text-cyan" />
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
              Model card
            </p>
            <h2 className="font-display text-lg font-semibold">
              {card.model_type} v{card.model_version ?? '--'}
            </h2>
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-4">
          <CardStat label="Policy" value={card.default_execution_policy.replace('_', '-')} />
          <CardStat label="Margin" value={formatProbability(card.default_margin)} />
          <CardStat label="Rows" value={formatInteger(numberValue(card.training_data.row_count))} />
          <CardStat label="Features" value={formatInteger(numberValue(card.feature_set.feature_count))} />
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {card.evaluation_splits.map((split) => {
            const tradeCount = split.metrics.trade_count?.value
            const roi = split.metrics.roi_on_cost?.value
            const brier = split.metrics.brier?.value
            const totalCost = split.metrics.total_cost?.value
            const evPer10 =
              tradeCount && roi !== undefined && totalCost !== undefined
                ? (roi * totalCost * 10) / tradeCount
                : null
            return (
              <div key={split.name} className="rounded-md border border-line bg-background/55 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-mono text-sm font-semibold text-foreground">{split.name}</p>
                    <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
                      {split.role} / {split.policy}
                    </p>
                  </div>
                  <span className="rounded border border-line px-2 py-1 font-mono text-[10px] text-muted">
                    {formatInteger(split.market_count)} mkts
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-2">
                  <SmallStat label="Trades" value={formatInteger(tradeCount)} />
                  <SmallStat label="ROI" value={formatProbability(roi)} tone="text-green" />
                  <SmallStat label="Brier" value={brier?.toFixed(4) ?? '--'} />
                  <SmallStat label="EV/10" value={formatSigned(evPer10)} tone="text-green" />
                </div>
                {split.notes ? <p className="mt-3 text-xs leading-5 text-muted">{split.notes}</p> : null}
              </div>
            )
          })}
        </div>
      </div>

      <aside className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">Caveats</p>
        <ul className="mt-3 space-y-2 text-sm leading-6 text-muted">
          {(card.caveats ?? ['No model-card caveats provided.']).map((caveat) => (
            <li key={caveat} className="border-l border-line pl-3">
              {caveat}
            </li>
          ))}
        </ul>
        <details className="mt-4 rounded-md border border-line bg-background/65 p-3">
          <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.16em] text-cyan">
            Full raw card JSON
          </summary>
          <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-muted">
            {JSON.stringify(card, null, 2)}
          </pre>
        </details>
      </aside>
    </section>
  )
}

function LegacyMetadataNotice({ model }: { readonly model: SavedModelMetadata | null }) {
  return (
    <div className="rounded-lg border border-amber/35 bg-amber/10 p-4 text-sm leading-6 text-amber">
      {model
        ? 'This model does not expose artifacts/model-card.json yet, so the overview is using legacy bundle metadata.'
        : 'No saved model is currently selected.'}
    </div>
  )
}

function numberValue(value: number | string | string[] | null | undefined): number | null {
  return typeof value === 'number' ? value : null
}

function CardStat({
  label,
  value,
}: {
  readonly label: string
  readonly value: string
}) {
  return (
    <div className="rounded-md border border-line bg-background/60 p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted">{label}</p>
      <p className="mt-2 font-mono text-lg font-semibold text-foreground">{value}</p>
    </div>
  )
}

function SmallStat({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div>
      <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-muted">{label}</p>
      <p className={`mt-1 font-mono text-sm font-semibold ${tone}`}>{value}</p>
    </div>
  )
}
