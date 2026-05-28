import { FileJson2 } from 'lucide-react'
import { useState } from 'react'

import { ExecutionModeControl } from '@/components/ExecutionModeControl'
import { ModelMetrics } from '@/components/ModelMetrics'
import { PollPredictionTable } from '@/components/PollPredictionTable'
import { PredictionTable } from '@/components/PredictionTable'
import { RiskReturnBandChart } from '@/components/RiskReturnBandChart'
import { ScoringPanel } from '@/components/ScoringPanel'
import { useRiskPresetTrials } from '@/hooks/useRiskPresetTrials'
import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type {
  ExecutionMode,
  PollPredictionRow,
  RiskPreset,
  RiskPresetTrial,
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
  readonly riskTrial: RiskPresetTrial | null
  readonly riskTrialLoading: boolean
  readonly riskPresets: readonly RiskPreset[]
  readonly selectedRiskPresetId: string | null
  readonly onSelectRiskPreset: (presetId: string) => void
  readonly onCreateRiskPreset: (preset: RiskPreset) => void
  readonly onUpdateRiskPreset: (preset: RiskPreset) => void
  readonly onDeleteRiskPreset: (presetId: string) => void
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
  riskTrial,
  riskTrialLoading,
  riskPresets,
  selectedRiskPresetId,
  onSelectRiskPreset,
  onCreateRiskPreset,
  onUpdateRiskPreset,
  onDeleteRiskPreset,
  onRowIndexChange,
  onExecutionModeChange,
  onScoreSample,
  onScoreUpload,
}: ModelOverviewPageProps) {
  const card = model?.model_card
  const riskTrials = useRiskPresetTrials(model, riskPresets)

  return (
    <section className="space-y-4">
      {card ? (
        <ModelCardPanel
          model={model}
          riskPreset={riskPreset}
          selectedTrial={riskTrial}
          riskTrialLoading={riskTrialLoading}
        />
      ) : (
        <LegacyMetadataNotice model={model} />
      )}

      <RiskReturnBandChart trial={riskTrial} loading={riskTrialLoading} />

      <RiskPresetWorkbench
        presets={riskPresets}
        trialsByPresetId={riskTrials.trialsByPresetId}
        loadingIds={riskTrials.loadingIds}
        error={riskTrials.error}
        selectedPresetId={selectedRiskPresetId}
        onSelect={onSelectRiskPreset}
        onCreate={onCreateRiskPreset}
        onUpdate={onUpdateRiskPreset}
        onDelete={onDeleteRiskPreset}
      />

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
  selectedTrial,
  riskTrialLoading,
}: {
  readonly model: SavedModelMetadata | null
  readonly riskPreset: RiskPreset | null
  readonly selectedTrial: RiskPresetTrial | null
  readonly riskTrialLoading: boolean
}) {
  const card = model?.model_card
  if (!card) return null
  const splits = card.evaluation_splits.filter((split) => !isFullScoredWindow(split))

  return (
    <section className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(22rem,0.7fr)]">
      <div className="grid gap-3 md:grid-cols-4 xl:col-span-2">
        <CardStat label="EV/10 markets" value={trialValue(formatSigned(selectedTrial?.ev_per_10_markets), riskTrialLoading)} />
        <CardStat label="Trade %" value={trialValue(formatProbability(selectedTrial?.trade_percent), riskTrialLoading)} />
        <CardStat
          label="Risk of ruin"
          value={trialValue(formatProbability(selectedTrial?.risk_of_ruin_estimate), riskTrialLoading)}
        />
        <CardStat
          label="Expected / market"
          value={trialValue(formatSigned(selectedTrial?.expected_return_per_market.expected), riskTrialLoading)}
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

      <RiskPresetPanel
        riskPreset={riskPreset}
        selectedTrial={selectedTrial}
        riskTrialLoading={riskTrialLoading}
      />
    </section>
  )
}

function RiskPresetPanel({
  riskPreset,
  selectedTrial,
  riskTrialLoading,
}: {
  readonly riskPreset: RiskPreset | null
  readonly selectedTrial: RiskPresetTrial | null
  readonly riskTrialLoading: boolean
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
          value={trialValue(formatProbability(selectedTrial?.risk_of_ruin_estimate), riskTrialLoading)}
        />
        <CardStat
          label="EV/10 markets"
          value={trialValue(formatSigned(selectedTrial?.ev_per_10_markets), riskTrialLoading)}
        />
        <CardStat label="Trade %" value={trialValue(formatProbability(selectedTrial?.trade_percent), riskTrialLoading)} />
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
        {riskTrialLoading ? (
          <p className="mt-2 text-xs leading-5 text-muted">
            Computing this model+preset trial from saved evaluation rows...
          </p>
        ) : selectedTrial ? (
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

function RiskPresetWorkbench({
  presets,
  trialsByPresetId,
  loadingIds,
  error,
  selectedPresetId,
  onSelect,
  onCreate,
  onUpdate,
  onDelete,
}: {
  readonly presets: readonly RiskPreset[]
  readonly trialsByPresetId: ReadonlyMap<string, RiskPresetTrial>
  readonly loadingIds: ReadonlySet<string>
  readonly error: string | null
  readonly selectedPresetId: string | null
  readonly onSelect: (presetId: string) => void
  readonly onCreate: (preset: RiskPreset) => void
  readonly onUpdate: (preset: RiskPreset) => void
  readonly onDelete: (presetId: string) => void
}) {
  const selected = presets.find((preset) => preset.id === selectedPresetId) ?? presets[0] ?? null
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState<RiskPreset | null>(null)

  const startEditing = (preset: RiskPreset) => {
    setEditingId(preset.id)
    setDraft(preset)
  }
  const cancelEditing = () => {
    setEditingId(null)
    setDraft(null)
  }
  const updateDraft = <K extends keyof RiskPreset>(key: K, value: RiskPreset[K]) => {
    setDraft((current) => (current ? { ...current, [key]: value } : current))
  }
  const saveDraft = () => {
    if (!draft) return
    onUpdate({
      ...draft,
      label: draft.label.trim() || 'Custom preset',
      description: draft.description.trim() || 'Custom risk preset',
    })
    cancelEditing()
  }
  const cloneSelected = () => {
    const base = selected ?? presets[0]
    if (!base) return
    onCreate({
      ...base,
      id: uniquePresetId(`${base.label} Copy`, presets),
      label: `${base.label} Copy`,
      description: base.description || 'Custom risk preset',
    })
  }

  return (
    <section className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Risk preset research desk
          </p>
          <h2 className="font-display text-lg font-semibold">Preset policies and expected risk</h2>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">
            Compare each sizing policy against the selected model. Custom rows are saved to the
            local workstation preset file; trial metrics compute from saved evaluation rows when
            a bundled result is not available.
          </p>
        </div>
        <button
          type="button"
          onClick={cloneSelected}
          className="rounded border border-cyan/40 bg-cyan/10 px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan transition hover:border-cyan"
        >
          Create preset
        </button>
      </div>
      {error ? (
        <div className="mb-3 rounded-md border border-amber/35 bg-amber/10 px-3 py-2 text-sm text-amber">
          {error}
        </div>
      ) : null}
      <div className="overflow-x-auto rounded-lg border border-line bg-background/45">
        <table className="w-full min-w-[1180px] border-collapse text-left text-xs">
          <thead className="border-b border-line bg-panelStrong/90 font-mono text-[9px] uppercase tracking-[0.16em] text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">Preset</th>
              <th className="px-3 py-2 font-medium">Policy</th>
              <th className="px-3 py-2 text-right font-medium">Risk of ruin</th>
              <th className="px-3 py-2 text-right font-medium">Return / mkt</th>
              <th className="px-3 py-2 text-right font-medium">EV / 10</th>
              <th className="px-3 py-2 text-right font-medium">Variance</th>
              <th className="px-3 py-2 text-right font-medium">Trade %</th>
              <th className="px-3 py-2 text-right font-medium">Markets</th>
              <th className="px-3 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/70">
            {presets.map((preset) => {
              const trial = trialsByPresetId.get(preset.id)
              const loading = loadingIds.has(preset.id)
              const editing = editingId === preset.id && draft
              return (
                <tr
                  key={preset.id}
                  className={preset.id === selectedPresetId ? 'bg-cyan/5' : 'hover:bg-panelStrong/35'}
                >
                  <td className="max-w-[20rem] px-3 py-2">
                    {editing ? (
                      <div className="grid gap-2">
                        <input
                          value={draft.label}
                          onChange={(event) => updateDraft('label', event.target.value)}
                          className="h-8 rounded-md border border-line bg-background px-2 font-mono text-xs"
                        />
                        <input
                          value={draft.description}
                          onChange={(event) => updateDraft('description', event.target.value)}
                          className="h-8 rounded-md border border-line bg-background px-2 text-xs"
                        />
                      </div>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => onSelect(preset.id)}
                          className="text-left font-mono text-xs font-semibold text-foreground transition hover:text-cyan"
                        >
                          {preset.label}
                        </button>
                        <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted">{preset.description}</p>
                      </>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    {editing ? (
                      <div className="grid grid-cols-5 gap-2">
                        <select
                          value={draft.trade_side}
                          onChange={(event) =>
                            updateDraft('trade_side', event.target.value as RiskPreset['trade_side'])
                          }
                          className="h-8 rounded-md border border-line bg-background px-2 font-mono text-xs"
                        >
                          <option value="all">All</option>
                          <option value="no_only">NO</option>
                          <option value="yes_only">YES</option>
                        </select>
                        <PercentInput value={draft.min_margin} onChange={(value) => updateDraft('min_margin', value)} />
                        <PercentInput value={draft.kelly_fraction} onChange={(value) => updateDraft('kelly_fraction', value)} />
                        <PercentInput value={draft.max_position_fraction} onChange={(value) => updateDraft('max_position_fraction', value)} />
                        <PercentInput value={draft.max_event_exposure_fraction} onChange={(value) => updateDraft('max_event_exposure_fraction', value)} />
                      </div>
                    ) : (
                      <div className="grid grid-cols-5 gap-2 font-mono text-[10px] uppercase tracking-[0.1em] text-muted">
                        <span>{preset.trade_side.replace('_', '-')}</span>
                        <span>M {formatProbability(preset.min_margin)}</span>
                        <span>K {formatProbability(preset.kelly_fraction)}</span>
                        <span>P {formatProbability(preset.max_position_fraction)}</span>
                        <span>E {formatProbability(preset.max_event_exposure_fraction)}</span>
                      </div>
                    )}
                  </td>
                  <RiskMetricCell value={loading ? 'Computing' : formatProbability(trial?.risk_of_ruin_estimate)} tone={ruinTone(trial)} />
                  <RiskMetricCell value={loading ? 'Computing' : formatSigned(trial?.expected_return_per_market.expected)} />
                  <RiskMetricCell value={loading ? 'Computing' : formatSigned(trial?.ev_per_10_markets)} />
                  <RiskMetricCell value={loading ? 'Computing' : formatVariance(trial?.return_variance_per_market)} />
                  <RiskMetricCell value={loading ? 'Computing' : formatProbability(trial?.trade_percent)} />
                  <RiskMetricCell value={loading ? 'Computing' : formatInteger(trial?.market_count)} />
                  <td className="px-3 py-2 text-right">
                    {editing ? (
                      <div className="flex justify-end gap-2">
                        <button type="button" onClick={saveDraft} className="text-cyan hover:text-foreground">
                          Save
                        </button>
                        <button type="button" onClick={cancelEditing} className="text-muted hover:text-foreground">
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div className="flex justify-end gap-2">
                        <button type="button" onClick={() => startEditing(preset)} className="text-cyan hover:text-foreground">
                          Edit
                        </button>
                        <button
                          type="button"
                          disabled={presets.length <= 1}
                          onClick={() => onDelete(preset.id)}
                          className="text-red hover:text-foreground disabled:cursor-not-allowed disabled:text-muted"
                        >
                          Delete
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function PercentInput({
  value,
  onChange,
}: {
  readonly value: number
  readonly onChange: (value: number) => void
}) {
  return (
    <input
      type="number"
      min="0"
      step="0.1"
      value={toPercentInput(value)}
      onChange={(event) => onChange(fromPercentInput(event.target.value))}
      className="h-8 min-w-0 rounded-md border border-line bg-background px-2 text-right font-mono text-xs"
    />
  )
}

function RiskMetricCell({
  value,
  tone = 'text-foreground',
}: {
  readonly value: string
  readonly tone?: string
}) {
  return <td className={`px-3 py-2 text-right font-mono text-xs font-semibold ${tone}`}>{value}</td>
}

function ruinTone(trial: RiskPresetTrial | undefined): string {
  if (!trial) return 'text-muted'
  if (trial.risk_of_ruin_estimate <= 0.01) return 'text-green'
  if (trial.risk_of_ruin_estimate <= 0.05) return 'text-cyan'
  if (trial.risk_of_ruin_estimate <= 0.15) return 'text-amber'
  return 'text-red'
}

function formatVariance(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(6) : '--'
}

function uniquePresetId(label: string, presets: readonly RiskPreset[]): string {
  const slug = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'custom-preset'
  const existing = new Set(presets.map((preset) => preset.id))
  let candidate = slug
  let suffix = 2
  while (existing.has(candidate)) {
    candidate = `${slug}-${suffix}`
    suffix += 1
  }
  return candidate
}

function toPercentInput(value: number): string {
  return Number.isFinite(value) ? String(Number((value * 100).toFixed(3))) : '0'
}

function fromPercentInput(value: string): number {
  const parsed = Number.parseFloat(value)
  if (!Number.isFinite(parsed)) return 0
  return parsed / 100
}

function trialValue(value: string, loading: boolean): string {
  return loading ? 'Computing...' : value
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
