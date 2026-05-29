import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'

import { AutoTraderPage } from '@/components/AutoTraderPage'
import { CurrentMarketsPage } from '@/components/CurrentMarketsPage'
import { ModelOverviewPage } from '@/components/ModelOverviewPage'
import { ModelPickerDropdown } from '@/components/ModelPickerDropdown'
import { RiskPresetDropdown } from '@/components/RiskPresetDropdown'
import { TradingHistoryPage } from '@/components/TradingHistoryPage'
import { useAccountSummary } from '@/hooks/useAccountSummary'
import { useLiveCurrentMarkets } from '@/hooks/useLiveCurrentMarkets'
import { useOpenPositions } from '@/hooks/useOpenPositions'
import { useRiskPresetTrial } from '@/hooks/useRiskPresetTrial'
import { useTrader } from '@/hooks/useTrader'
import { useWorkstation } from '@/hooks/useWorkstation'
import {
  createRiskPreset as createRiskPresetFile,
  deleteRiskPreset as deleteRiskPresetFile,
  listRiskPresets,
  updateRiskPreset as updateRiskPresetFile,
} from '@/lib/api'
import { formatDollars } from '@/lib/format'
import type { AccountSummary, RiskPreset } from '@/lib/types'

type ViewId = 'markets' | 'history' | 'model' | 'trader'

function App() {
  const workstation = useWorkstation()
  const account = useAccountSummary()
  const openPositions = useOpenPositions()
  const [riskPresets, setRiskPresets] = useState<RiskPreset[]>([])
  const [selectedRiskPresetId, setSelectedRiskPresetId] = useState<string | null>(null)
  const [riskPresetError, setRiskPresetError] = useState<string | null>(null)
  const selectedRiskPreset =
    riskPresets.find((preset) => preset.id === selectedRiskPresetId) ?? riskPresets[0] ?? null
  const currentMarkets = useLiveCurrentMarkets(workstation.selectedModelName, selectedRiskPreset)
  const riskTrial = useRiskPresetTrial(workstation.selectedModel, selectedRiskPreset)
  const trader = useTrader({
    stagedModelName: workstation.selectedModelName,
    stagedRiskPreset: selectedRiskPreset,
  })
  const [activeView, setActiveView] = useState<ViewId>('markets')

  useEffect(() => {
    let cancelled = false
    listRiskPresets()
      .then((presets) => {
        if (cancelled) return
        setRiskPresets(presets)
        setSelectedRiskPresetId((current) => current ?? 'balanced')
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setRiskPresetError(err instanceof Error ? err.message : 'Failed to load risk presets')
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const createRiskPreset = async (preset: RiskPreset) => {
    try {
      const presets = await createRiskPresetFile(preset)
      setRiskPresets(presets)
      setSelectedRiskPresetId(preset.id)
      setRiskPresetError(null)
    } catch (err) {
      setRiskPresetError(err instanceof Error ? err.message : 'Failed to save risk preset')
    }
  }

  const updateRiskPreset = async (preset: RiskPreset) => {
    try {
      const presets = await updateRiskPresetFile(preset)
      setRiskPresets(presets)
      setSelectedRiskPresetId(preset.id)
      setRiskPresetError(null)
    } catch (err) {
      setRiskPresetError(err instanceof Error ? err.message : 'Failed to update risk preset')
    }
  }

  const deleteRiskPreset = async (presetId: string) => {
    try {
      const presets = await deleteRiskPresetFile(presetId)
      setRiskPresets(presets)
      setSelectedRiskPresetId((current) => {
        if (current && presets.some((preset) => preset.id === current)) return current
        return presets[0]?.id ?? null
      })
      setRiskPresetError(null)
    } catch (err) {
      setRiskPresetError(err instanceof Error ? err.message : 'Failed to delete risk preset')
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen w-full flex-col bg-background/88">
        <header className="border-b border-line bg-background shadow-terminal">
          <div className="flex min-h-12 flex-wrap items-center gap-2 px-3 py-1.5">
            <div className="flex min-w-0 shrink-0 flex-wrap items-center gap-2">
              <div className="font-mono text-xs font-semibold uppercase tracking-[0.22em] text-cyan">
                Kalorie
              </div>
              <nav className="flex flex-wrap items-center gap-1 rounded-md border border-line bg-panel/58 p-0.5">
                <ViewButton
                  active={activeView === 'markets'}
                  label="Current Markets"
                  onClick={() => setActiveView('markets')}
                />
                <ViewButton
                  active={activeView === 'history'}
                  label="Trading History"
                  onClick={() => setActiveView('history')}
                />
                <ViewButton
                  active={activeView === 'model'}
                  label="Model Overview"
                  onClick={() => setActiveView('model')}
                />
                <ViewButton
                  active={activeView === 'trader'}
                  label="Auto Trader"
                  onClick={() => setActiveView('trader')}
                />
              </nav>
            </div>
            <div className="grid min-w-[24rem] flex-1 grid-cols-[minmax(11rem,1fr)_minmax(11rem,1fr)] gap-2">
              <ModelPickerDropdown
                models={workstation.models}
                selectedModelName={workstation.selectedModelName}
                onSelect={workstation.selectModel}
                compact
              />
              <RiskPresetDropdown
                presets={riskPresets}
                selectedPresetId={selectedRiskPreset?.id ?? null}
                onSelect={setSelectedRiskPresetId}
                compact
              />
            </div>
            <div className="ml-auto flex shrink-0 flex-wrap items-center gap-3">
              <PortfolioReadout summary={account.summary} />
              <TraderBadge
                running={trader.status?.running ?? false}
                mode={trader.status?.mode ?? 'off'}
                onClick={() => setActiveView('trader')}
              />
            </div>
          </div>
        </header>

        <main className="space-y-3 p-3">
          {workstation.loading ? (
            <SystemNotice tone="cyan" message="Scanning top-level models/* for valid saved bundles..." />
          ) : null}
          {workstation.error ? <SystemNotice tone="amber" message={workstation.error} /> : null}
          {riskPresetError ? <SystemNotice tone="amber" message={riskPresetError} /> : null}
          {currentMarkets.error ? <SystemNotice tone="amber" message={currentMarkets.error} /> : null}
          {account.error && account.summary.source !== 'paper' ? (
            <SystemNotice tone="amber" message={account.error} />
          ) : null}
          {openPositions.error && openPositions.error !== 'Kalshi account auth is not configured' ? (
            <SystemNotice tone="amber" message={openPositions.error} />
          ) : null}
          {riskTrial.error ? <SystemNotice tone="amber" message={riskTrial.error} /> : null}

          {activeView === 'markets' ? (
            <CurrentMarketsPage
              snapshot={currentMarkets.snapshot}
              loading={currentMarkets.loading}
              streamStatus={currentMarkets.streamStatus}
              bankroll={account.summary.bankroll}
              onRefresh={currentMarkets.refresh}
            />
          ) : null}

          {activeView === 'history' ? (
            <TradingHistoryPage
              positions={openPositions.summary}
              currentMarketRows={currentMarkets.snapshot?.prediction_rows ?? []}
            />
          ) : null}

          {activeView === 'trader' ? (
            <AutoTraderPage
              status={trader.status}
              activity={trader.activity}
              error={trader.error}
              busy={trader.busy}
              stagedModelName={workstation.selectedModelName}
              stagedRiskPreset={selectedRiskPreset}
              stagedDiffersFromRunning={trader.stagedDiffersFromRunning}
              positions={openPositions.summary}
              onStart={() => void trader.start()}
              onStop={() => void trader.stop()}
              onRestart={() => void trader.restart()}
              onKill={() => void trader.kill()}
              onResume={() => void trader.resume()}
            />
          ) : null}

          {activeView === 'model' ? (
            <ModelOverviewPage
              model={workstation.selectedModel}
              riskPreset={selectedRiskPreset}
              sampleRows={workstation.sampleRows}
              selectedRowIndex={workstation.selectedRowIndex}
              executionMode={workstation.executionMode}
              predictions={workstation.predictions}
              scoring={workstation.scoring}
              currentMarketRows={currentMarkets.snapshot?.prediction_rows ?? []}
              currentMarketsLoading={currentMarkets.loading}
              riskTrial={riskTrial.trial}
              riskTrialLoading={riskTrial.loading}
              riskPresets={riskPresets}
              selectedRiskPresetId={selectedRiskPreset?.id ?? null}
              onSelectRiskPreset={setSelectedRiskPresetId}
              onCreateRiskPreset={createRiskPreset}
              onUpdateRiskPreset={updateRiskPreset}
              onDeleteRiskPreset={deleteRiskPreset}
              onRowIndexChange={workstation.setSelectedRowIndex}
              onExecutionModeChange={workstation.setExecutionMode}
              onScoreSample={workstation.scoreSelectedRow}
              onScoreUpload={workstation.scoreUploadedCsv}
            />
          ) : null}
        </main>
      </div>
    </div>
  )
}

interface SystemNoticeProps {
  readonly tone: 'cyan' | 'amber'
  readonly message: string
}

function SystemNotice({ tone, message }: SystemNoticeProps) {
  const toneClass =
    tone === 'cyan' ? 'border-cyan/30 bg-cyan/10 text-cyan' : 'border-amber/30 bg-amber/10 text-amber'
  return (
    <div className={`flex items-center gap-2 rounded-xl border px-4 py-3 text-sm ${toneClass}`}>
      <AlertTriangle size={16} />
      {message}
    </div>
  )
}

function PortfolioReadout({ summary }: { readonly summary: AccountSummary }) {
  const portfolioLabel =
    summary.available && summary.portfolio_value !== null && summary.portfolio_value !== undefined
      ? formatDollars(summary.portfolio_value)
      : summary.source === 'paper'
        ? 'Paper bankroll'
        : '--'
  const freeCashLabel =
    summary.available && summary.free_cash !== null && summary.free_cash !== undefined
      ? `Free ${formatDollars(summary.free_cash)}`
      : `Sizing ${formatDollars(summary.bankroll)}`
  return (
    <div className="flex h-8 items-center gap-2 font-mono">
      <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Portfolio
      </span>
      <span className="font-mono text-sm font-semibold text-foreground">{portfolioLabel}</span>
      <span className="font-mono text-xs text-muted">{freeCashLabel}</span>
    </div>
  )
}

interface TraderBadgeProps {
  readonly running: boolean
  readonly mode: string
  readonly onClick: () => void
}

function TraderBadge({ running, mode, onClick }: TraderBadgeProps) {
  const live = mode === 'live'
  const toneClass = !running
    ? 'border-line bg-panelStrong text-muted'
    : live
      ? 'border-red/35 bg-red/10 text-red'
      : 'border-green/35 bg-green/10 text-green'
  const dotClass = !running ? 'bg-muted' : live ? 'bg-red' : 'bg-green'
  const label = !running ? 'Trader off' : live ? 'Live trading' : 'Dry run'
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex h-8 items-center gap-2 rounded-lg border px-3 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] transition ${toneClass}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass} ${running ? 'animate-pulse' : ''}`} />
      {label}
    </button>
  )
}

interface ViewButtonProps {
  readonly active: boolean
  readonly label: string
  readonly onClick: () => void
}

function ViewButton({ active, label, onClick }: ViewButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'rounded px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] transition',
        active
          ? 'border-cyan/70 bg-panelStrong text-cyan shadow-bloom'
          : 'border-transparent text-muted hover:bg-panel hover:text-foreground',
      ].join(' ')}
    >
      {label}
    </button>
  )
}

export default App
