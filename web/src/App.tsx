import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'

import { CurrentMarketsPage } from '@/components/CurrentMarketsPage'
import { ModelOverviewPage } from '@/components/ModelOverviewPage'
import { ModelPickerDropdown } from '@/components/ModelPickerDropdown'
import { RiskPresetDropdown } from '@/components/RiskPresetDropdown'
import { TradingHistoryPage } from '@/components/TradingHistoryPage'
import { useAccountSummary } from '@/hooks/useAccountSummary'
import { useLiveCurrentMarkets } from '@/hooks/useLiveCurrentMarkets'
import { useOpenPositions } from '@/hooks/useOpenPositions'
import { useRiskPresetTrial } from '@/hooks/useRiskPresetTrial'
import { useWorkstation } from '@/hooks/useWorkstation'
import { listRiskPresets } from '@/lib/api'
import { formatDollars } from '@/lib/format'
import type { AccountSummary, RiskPreset } from '@/lib/types'

type ViewId = 'markets' | 'history' | 'model'

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

  const createRiskPreset = (preset: RiskPreset) => {
    setRiskPresets((current) => {
      const withoutDuplicate = current.filter((entry) => entry.id !== preset.id)
      return [...withoutDuplicate, preset]
    })
    setSelectedRiskPresetId(preset.id)
  }

  const deleteRiskPreset = (presetId: string) => {
    setRiskPresets((current) => {
      if (current.length <= 1) return current
      return current.filter((preset) => preset.id !== presetId)
    })
    setSelectedRiskPresetId((current) => {
      if (current !== presetId) return current
      return riskPresets.find((preset) => preset.id !== presetId)?.id ?? null
    })
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1500px] flex-col bg-background/72">
        <header className="border-b border-line bg-panel/78 px-4 py-4 shadow-terminal">
          <div className="flex flex-col gap-4 2xl:flex-row 2xl:items-end 2xl:justify-between">
            <div className="flex flex-col gap-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-cyan">
                Kalorie institutional terminal
              </p>
              <h1 className="mt-2 font-display text-2xl font-semibold tracking-tight">
                Earnings mention market workstation
              </h1>
              <PortfolioReadout summary={account.summary} />
            </div>
            <div className="grid gap-3 xl:grid-cols-2">
              <ModelPickerDropdown
                models={workstation.models}
                selectedModelName={workstation.selectedModelName}
                onSelect={workstation.selectModel}
              />
              <RiskPresetDropdown
                presets={riskPresets}
                selectedPresetId={selectedRiskPreset?.id ?? null}
                onSelect={setSelectedRiskPresetId}
                onCreate={createRiskPreset}
                onDelete={deleteRiskPreset}
              />
            </div>
          </div>
          <nav className="mt-4 grid rounded-xl border border-line bg-background/65 p-1 md:grid-cols-3">
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
          </nav>
        </header>

        <main className="space-y-4 p-4">
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
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-background/60 px-3 py-2">
      <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Portfolio
      </span>
      <span className="font-mono text-lg font-semibold text-foreground">{portfolioLabel}</span>
      <span className="font-mono text-xs text-muted">{freeCashLabel}</span>
    </div>
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
        'rounded-lg border px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.16em] transition',
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
