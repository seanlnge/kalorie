import { FileJson2 } from 'lucide-react'

import { ModelMetrics } from '@/components/ModelMetrics'
import { PollPredictionTable } from '@/components/PollPredictionTable'
import { RiskReturnBandChart } from '@/components/RiskReturnBandChart'
import { ScoringPanel } from '@/components/ScoringPanel'
import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow, RiskPreset, SampleRow, SavedModelMetadata } from '@/lib/types'

export interface ModelOverviewPageProps {
  readonly model: SavedModelMetadata | null
  readonly riskPreset: RiskPreset | null
  readonly sampleRows: readonly SampleRow[]
  readonly selectedRowIndex: number
  readonly scoring: boolean
  readonly currentMarketRows: readonly PollPredictionRow[]
  readonly currentMarketsLoading: boolean
  readonly onRowIndexChange: (rowIndex: number) => void
  readonly onScoreSample: () => Promise<void>
  readonly onScoreUpload: (file: File) => Promise<void>
}

export function ModelOverviewPage({
  model,
  riskPreset,
  sampleRows,
  selectedRowIndex,
  scoring,
  currentMarketRows,
  currentMarketsLoading,
  onRowIndexChange,
  onScoreSample,
  onScoreUpload,
}: ModelOverviewPageProps) {
  const card = model?.model_card

  return (
    <section className="space-y-4">
      {card ? (
        <ModelCardPanel model={model} riskPreset={riskPreset} />
      ) : (
        <LegacyMetadataNotice model={model} />
      )}

      {model?.risk_preset_trials.length ? (
        <RiskReturnBandChart
          trials={model.risk_preset_trials}
          selectedRiskPresetId={riskPreset?.id ?? null}
        />
      ) : null}

      <ModelMetrics model={model} />

      <ScoringPanel
        sampleRows={sampleRows}
        selectedRowIndex={selectedRowIndex}
        scoring={scoring}
        onRowIndexChange={onRowIndexChange}
        onScoreSample={onScoreSample}
        onScoreUpload={onScoreUpload}
      />
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
          loading={currentMarketsLoading}
          emptyMessage="No active-market scores for the selected model yet."
        />
      </section>
    </section>
  )
}

function ModelCardPanel({
  model,
  riskPreset,
}: {
  readonly model: SavedModelMetadata | null
  readonly riskPreset: RiskPreset | null
}) {
  const card = model?.model_card
  if (!card) return null
  const splits = card.evaluation_splits.filter((split) => !isFullScoredWindow(split))
  const selectedTrial =
    model?.risk_preset_trials.find((trial) => trial.risk_preset_id === riskPreset?.id) ??
    model?.risk_preset_trials[0] ??
    null

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
          <CardStat label="Rows" value={formatInteger(numberValue(card.training_data.row_count))} />
          <CardStat label="Events" value={formatInteger(numberValue(card.training_data.event_count))} />
          <CardStat label="Features" value={formatInteger(numberValue(card.feature_set.feature_count))} />
          <CardStat label="Markets" value={formatInteger(splits[0]?.market_count)} />
        </div>
        {selectedTrial ? (
          <div className="mt-4 grid gap-3 md:grid-cols-5">
            <CardStat label="Risk preset" value={riskPreset?.label ?? selectedTrial.label} />
            <CardStat label="Min margin" value={formatProbability(selectedTrial.min_margin)} />
            <CardStat label="Trade %" value={formatProbability(selectedTrial.trade_percent)} />
            <CardStat label="EV/10 markets" value={formatSigned(selectedTrial.ev_per_10_markets)} />
            <CardStat label="Risk of ruin" value={selectedTrial.risk_of_ruin_label} />
          </div>
        ) : null}
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {splits.map((split) => {
            const brier = split.metrics.brier?.value
            const marketBrier = split.metrics.market_brier?.value
            const ece = split.metrics.ece?.value
            const logLoss = split.metrics.log_loss?.value
            return (
              <div key={split.name} className="rounded-md border border-line bg-background/55 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-mono text-sm font-semibold text-foreground">
                      {displaySplitName(split.name)}
                    </p>
                    <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
                      {split.role} / predictive quality
                    </p>
                  </div>
                  <span className="rounded border border-line px-2 py-1 font-mono text-[10px] text-muted">
                    {formatInteger(split.market_count)} mkts
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-2">
                  <SmallStat label="Brier" value={brier?.toFixed(4) ?? '--'} />
                  <SmallStat label="Market Brier" value={marketBrier?.toFixed(4) ?? '--'} />
                  <SmallStat label="ECE" value={ece?.toFixed(4) ?? '--'} />
                  <SmallStat label="Log loss" value={logLoss?.toFixed(4) ?? '--'} />
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

function isFullScoredWindow(split: { readonly name: string }): boolean {
  const normalized = split.name.toLowerCase()
  return normalized.includes('full') || normalized.includes('walk')
}

function displaySplitName(name: string): string {
  if (name.toLowerCase().includes('latest')) {
    return 'Testing suite results'
  }
  return name
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
