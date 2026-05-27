import { FileJson2 } from 'lucide-react'

import { ExecutionModeControl } from '@/components/ExecutionModeControl'
import { ModelMetrics } from '@/components/ModelMetrics'
import { PollPredictionTable } from '@/components/PollPredictionTable'
import { PredictionTable } from '@/components/PredictionTable'
import { RiskReturnBandChart } from '@/components/RiskReturnBandChart'
import { ScoringPanel } from '@/components/ScoringPanel'
import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type {
  ExecutionMode,
  PollPredictionRow,
  RiskPreset,
  SampleRow,
  SavedModelMetadata,
  ScoreRow,
} from '@/lib/types'

export interface ModelOverviewPageProps {
  readonly model: SavedModelMetadata | null
  readonly riskPreset: RiskPreset | null
  readonly sampleRows: readonly SampleRow[]
  readonly selectedRowIndex: number
  readonly executionMode: ExecutionMode
  readonly predictions: readonly ScoreRow[]
  readonly scoring: boolean
  readonly currentMarketRows: readonly PollPredictionRow[]
  readonly currentMarketsLoading: boolean
  readonly onRowIndexChange: (rowIndex: number) => void
  readonly onExecutionModeChange: (mode: ExecutionMode) => void
  readonly onScoreSample: () => Promise<void>
  readonly onScoreUpload: (file: File) => Promise<void>
}

export function ModelOverviewPage({
  model,
  riskPreset,
  sampleRows,
  selectedRowIndex,
  executionMode,
  predictions,
  scoring,
  currentMarketRows,
  currentMarketsLoading,
  onRowIndexChange,
  onExecutionModeChange,
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

      <RiskReturnBandChart trial={selectedRiskPresetTrial(model, riskPreset)} />

      <ModelMetrics model={model} />

      <ExecutionModeControl mode={executionMode} onChange={onExecutionModeChange} />
      <ScoringPanel
        sampleRows={sampleRows}
        selectedRowIndex={selectedRowIndex}
        scoring={scoring}
        disabled={!model}
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
  const selectedTrial = selectedRiskPresetTrial(model, riskPreset)

  return (
    <section className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(22rem,0.7fr)]">
      <div className="grid gap-3 md:grid-cols-4 xl:col-span-2">
        <CardStat
          label="EV/10 markets"
          value={formatSigned(selectedTrial?.ev_per_10_markets)}
        />
        <CardStat label="Trade %" value={formatProbability(selectedTrial?.trade_percent)} />
        <CardStat
          label="Risk of ruin"
          value={formatProbability(selectedTrial?.risk_of_ruin_estimate)}
        />
        <CardStat
          label="Expected / market"
          value={formatSigned(selectedTrial?.expected_return_per_market.expected)}
        />
      </div>
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
                {split.notes ? (
                  <p className="mt-3 text-xs leading-5 text-muted">
                    {displaySplitNotes(split.notes)}
                  </p>
                ) : null}
              </div>
            )
          })}
        </div>
      </div>

      <RiskPresetPanel riskPreset={riskPreset} selectedTrial={selectedTrial} />
    </section>
  )
}

function RiskPresetPanel({
  riskPreset,
  selectedTrial,
}: {
  readonly riskPreset: RiskPreset | null
  readonly selectedTrial: SavedModelMetadata['risk_preset_trials'][number] | null
}) {
  return (
    <aside className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
      <div className="mb-4">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          Risk preset overlay
        </p>
        <h2 className="font-display text-lg font-semibold">
          {riskPreset?.label ?? selectedTrial?.label ?? 'No preset selected'}
        </h2>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <CardStat label="Min margin" value={formatProbability(riskPreset?.min_margin)} />
        <CardStat label="Kelly" value={formatProbability(riskPreset?.kelly_fraction)} />
        <CardStat
          label="Position cap"
          value={formatProbability(riskPreset?.max_position_fraction)}
        />
        <CardStat
          label="Event cap"
          value={formatProbability(riskPreset?.max_event_exposure_fraction)}
        />
        <CardStat
          label="Risk of ruin"
          value={formatProbability(selectedTrial?.risk_of_ruin_estimate)}
        />
        <CardStat
          label="EV/10 markets"
          value={formatSigned(selectedTrial?.ev_per_10_markets)}
        />
        <CardStat label="Trade %" value={formatProbability(selectedTrial?.trade_percent)} />
        <CardStat label="Side policy" value={riskPreset?.trade_side.replace('_', '-') ?? '--'} />
      </div>
      <div className="mt-4 rounded-md border border-line bg-background/55 p-3">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
          Preset note
        </p>
        <p className="mt-2 text-sm leading-6 text-muted">
          {riskPreset?.description ??
            'Risk presets convert model probabilities into trade filters and sizing guidance.'}
        </p>
        {selectedTrial ? (
          <p className="mt-2 text-xs leading-5 text-muted">
            Risk of ruin is estimated from the selected model/preset trial bootstrap as the
            share of sampled event returns below zero:{' '}
            {formatProbability(selectedTrial.risk_of_ruin_estimate)} (
            {selectedTrial.risk_of_ruin_label}).
          </p>
        ) : null}
      </div>
    </aside>
  )
}

function selectedRiskPresetTrial(
  model: SavedModelMetadata | null,
  riskPreset: RiskPreset | null,
): SavedModelMetadata['risk_preset_trials'][number] | null {
  if (!model?.risk_preset_trials.length) {
    return null
  }
  if (!riskPreset) {
    return model.risk_preset_trials[0] ?? null
  }
  return model.risk_preset_trials.find((trial) => trial.risk_preset_id === riskPreset.id) ?? null
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

function displaySplitNotes(notes: string): string {
  return notes.replace(/latest\s*30/gi, 'testing suite')
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
