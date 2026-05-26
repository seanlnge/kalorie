import { AlertTriangle } from 'lucide-react'
import { useState } from 'react'

import { CurrentMarketsPage } from '@/components/CurrentMarketsPage'
import { ModelOverviewPage } from '@/components/ModelOverviewPage'
import { ModelPickerDropdown } from '@/components/ModelPickerDropdown'
import { TopStatusBar } from '@/components/TopStatusBar'
import { TradingHistoryPage } from '@/components/TradingHistoryPage'
import { usePollSnapshot } from '@/hooks/usePollSnapshot'
import { useWorkstation } from '@/hooks/useWorkstation'

type ViewId = 'markets' | 'history' | 'model'

function App() {
  const workstation = useWorkstation()
  const pollSnapshot = usePollSnapshot()
  const [activeView, setActiveView] = useState<ViewId>('markets')

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1500px] flex-col bg-background/72">
        <TopStatusBar selectedModel={workstation.selectedModel} pollSnapshot={pollSnapshot.snapshot} />
        <header className="border-b border-line bg-panel/78 px-4 py-4 shadow-terminal">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-cyan">
                Kalorie institutional terminal
              </p>
              <h1 className="mt-2 font-display text-2xl font-semibold tracking-tight">
                Earnings mention market workstation
              </h1>
            </div>
            <ModelPickerDropdown
              models={workstation.models}
              selectedModelName={workstation.selectedModelName}
              onSelect={workstation.selectModel}
            />
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
          {pollSnapshot.error ? <SystemNotice tone="amber" message={pollSnapshot.error} /> : null}

          {activeView === 'markets' ? (
            <CurrentMarketsPage
              snapshot={pollSnapshot.snapshot}
              loading={pollSnapshot.loading}
              onRefresh={() => void pollSnapshot.refresh()}
            />
          ) : null}

          {activeView === 'history' ? (
            <TradingHistoryPage history={pollSnapshot.history} />
          ) : null}

          {activeView === 'model' ? (
            <ModelOverviewPage
              model={workstation.selectedModel}
              sampleRows={workstation.sampleRows}
              selectedRowIndex={workstation.selectedRowIndex}
              scoring={workstation.scoring}
              predictions={workstation.predictions}
              currentMarketRows={
                pollSnapshot.snapshot?.model_name === workstation.selectedModelName
                  ? pollSnapshot.snapshot.prediction_rows
                  : []
              }
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
