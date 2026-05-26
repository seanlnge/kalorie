import { AlertTriangle } from 'lucide-react'
import { useState } from 'react'

import { ExecutionModeControl } from '@/components/ExecutionModeControl'
import { LiveModelPage } from '@/components/LiveModelPage'
import { ModelDetailsDrawer } from '@/components/ModelDetailsDrawer'
import { ModelMetrics } from '@/components/ModelMetrics'
import { ModelSidebar } from '@/components/ModelSidebar'
import { PredictionTable } from '@/components/PredictionTable'
import { ScoringPanel } from '@/components/ScoringPanel'
import { TopStatusBar } from '@/components/TopStatusBar'
import { TradesPage } from '@/components/TradesPage'
import { usePollSnapshot } from '@/hooks/usePollSnapshot'
import { useWorkstation } from '@/hooks/useWorkstation'

type ViewId = 'workstation' | 'live' | 'trades'

function App() {
  const workstation = useWorkstation()
  const pollSnapshot = usePollSnapshot()
  const [activeView, setActiveView] = useState<ViewId>('workstation')

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <ModelSidebar
        models={workstation.models}
        selectedModelName={workstation.selectedModelName}
        onSelect={workstation.selectModel}
      />
      <div className="min-w-0 flex-1">
        <TopStatusBar selectedModel={workstation.selectedModel} />
        <main className="space-y-5 p-6">
          <nav className="flex w-fit rounded-2xl border border-line/70 bg-panel/80 p-1">
            <ViewButton active={activeView === 'workstation'} label="Workbench" onClick={() => setActiveView('workstation')} />
            <ViewButton active={activeView === 'live'} label="Live Model" onClick={() => setActiveView('live')} />
            <ViewButton active={activeView === 'trades'} label="Trades" onClick={() => setActiveView('trades')} />
          </nav>
          {workstation.loading ? (
            <SystemNotice tone="cyan" message="Scanning top-level models/* for valid saved bundles..." />
          ) : null}
          {workstation.error ? <SystemNotice tone="amber" message={workstation.error} /> : null}
          {pollSnapshot.error ? <SystemNotice tone="amber" message={pollSnapshot.error} /> : null}

          {activeView === 'workstation' ? (
            <>
              <section className="rounded-[2rem] border border-line/70 bg-panelStrong/50 p-6 shadow-terminal">
                <div className="max-w-5xl">
                  <p className="text-xs uppercase tracking-[0.32em] text-cyan">Quant research terminal</p>
                  <h2 className="mt-2 font-display text-4xl font-semibold tracking-tight">
                    Saved-model scoring for Kalshi earnings mention markets.
                  </h2>
                  <p className="mt-4 max-w-3xl text-sm leading-6 text-muted">
                    Select a model bundle, inspect the fitted residual engine, score market rows,
                    and separate broad opportunities from the historically stronger NO-only slice.
                  </p>
                </div>
              </section>

              <ModelMetrics model={workstation.selectedModel} />
              <ExecutionModeControl
                mode={workstation.executionMode}
                onChange={workstation.setExecutionMode}
              />
              <ScoringPanel
                sampleRows={workstation.sampleRows}
                selectedRowIndex={workstation.selectedRowIndex}
                scoring={workstation.scoring}
                onRowIndexChange={workstation.setSelectedRowIndex}
                onScoreSample={workstation.scoreSelectedRow}
                onScoreUpload={workstation.scoreUploadedCsv}
              />
              <PredictionTable rows={workstation.predictions} />
              <ModelDetailsDrawer model={workstation.selectedModel} />
            </>
          ) : null}

          {activeView === 'live' ? (
            <LiveModelPage
              snapshot={pollSnapshot.snapshot}
              loading={pollSnapshot.loading}
              onRefresh={() => void pollSnapshot.refresh()}
            />
          ) : null}

          {activeView === 'trades' ? (
            <TradesPage snapshot={pollSnapshot.snapshot} trades={pollSnapshot.trades} />
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
  const toneClass = tone === 'cyan' ? 'border-cyan/30 bg-cyan/10 text-cyan' : 'border-amber/30 bg-amber/10 text-amber'
  return (
    <div className={`flex items-center gap-2 rounded-2xl border px-4 py-3 text-sm ${toneClass}`}>
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
        'rounded-xl px-5 py-2 text-sm font-semibold transition',
        active ? 'bg-cyan text-background' : 'text-muted hover:text-foreground',
      ].join(' ')}
    >
      {label}
    </button>
  )
}

export default App
