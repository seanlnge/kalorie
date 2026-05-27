import { AlertTriangle } from 'lucide-react'
import { useEffect, useState } from 'react'

import { CurrentMarketsPage } from '@/components/CurrentMarketsPage'
import { ModelOverviewPage } from '@/components/ModelOverviewPage'
import { ModelPickerDropdown } from '@/components/ModelPickerDropdown'
import { RiskPresetDropdown } from '@/components/RiskPresetDropdown'
import { TradingHistoryPage } from '@/components/TradingHistoryPage'
import { useLiveCurrentMarkets } from '@/hooks/useLiveCurrentMarkets'
import { usePollSnapshot } from '@/hooks/usePollSnapshot'
import { useWorkstation } from '@/hooks/useWorkstation'
import { listRiskPresets } from '@/lib/api'
import type { RiskPreset } from '@/lib/types'

type ViewId = 'markets' | 'history' | 'model'

function App() {
  const workstation = useWorkstation()
  const pollSnapshot = usePollSnapshot()
  const [riskPresets, setRiskPresets] = useState<RiskPreset[]>([])
  const [selectedRiskPresetId, setSelectedRiskPresetId] = useState<string | null>(null)
  const [riskPresetError, setRiskPresetError] = useState<string | null>(null)
  const selectedRiskPreset =
    riskPresets.find((preset) => preset.id === selectedRiskPresetId) ?? riskPresets[0] ?? null
  const currentMarkets = useLiveCurrentMarkets(workstation.selectedModelName, selectedRiskPreset)
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
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-cyan">
                Kalorie institutional terminal
              </p>
              <h1 className="mt-2 font-display text-2xl font-semibold tracking-tight">
                Earnings mention market workstation
              </h1>
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
          <nav className="mt-4 flex flex-wrap gap-2">
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
          {pollSnapshot.error ? <SystemNotice tone="amber" message={pollSnapshot.error} /> : null}
          {currentMarkets.error ? <SystemNotice tone="amber" message={currentMarkets.error} /> : null}

          {activeView === 'markets' ? (
            <CurrentMarketsPage
              snapshot={currentMarkets.snapshot}
              loading={currentMarkets.loading}
              onRefresh={currentMarkets.refresh}
            />
          ) : null}

          {activeView === 'history' ? (
            <TradingHistoryPage history={pollSnapshot.history} />
          ) : null}

          {activeView === 'model' ? (
            <ModelOverviewPage
              model={workstation.selectedModel}
              riskPreset={selectedRiskPreset}
              sampleRows={workstation.sampleRows}
              selectedRowIndex={workstation.selectedRowIndex}
              scoring={workstation.scoring}
              currentMarketRows={currentMarkets.snapshot?.prediction_rows ?? []}
              currentMarketsLoading={currentMarkets.loading}
              onRowIndexChange={workstation.setSelectedRowIndex}
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
        'rounded-md border px-4 py-2 font-mono text-xs uppercase tracking-[0.16em] transition',
        active
          ? 'border-cyan/70 bg-background text-cyan'
          : 'border-line bg-panel text-muted hover:border-muted/50 hover:text-foreground',
      ].join(' ')}
    >
      {label}
    </button>
  )
}

export default App
