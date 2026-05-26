import { Activity, Clock3, Database, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'

import type { PollSnapshot, SavedModelMetadata } from '@/lib/types'

export interface TopStatusBarProps {
  readonly selectedModel: SavedModelMetadata | null
  readonly pollSnapshot: PollSnapshot | null
}

export function TopStatusBar({ selectedModel, pollSnapshot }: TopStatusBarProps) {
  const timestamp = selectedModel?.trained_at
    ? new Date(selectedModel.trained_at).toLocaleString()
    : 'No model selected'
  const pollTimestamp = pollSnapshot ? new Date(pollSnapshot.completed_at).toLocaleString() : 'No poll cache'

  return (
    <header className="sticky top-0 z-10 border-b border-line bg-background/90 px-4 py-3 backdrop-blur-xl">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        <div className="min-w-0">
          <p className="font-mono text-[10px] uppercase tracking-[0.36em] text-cyan">Kalorie2</p>
          <h1 className="font-display text-xl font-bold tracking-tight">
            Earnings Mention Prediction Workstation
          </h1>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <ReadOnlyPill />
          <StatusPill
            icon={<Activity size={14} />}
            label="Active model"
            value={selectedModel?.name ?? '--'}
          />
          <StatusPill icon={<Clock3 size={14} />} label="Trained" value={timestamp} />
          <StatusPill icon={<Database size={14} />} label="Poll cache" value={pollTimestamp} />
        </div>
      </div>
    </header>
  )
}

interface StatusPillProps {
  readonly icon: ReactNode
  readonly label: string
  readonly value: string
}

function StatusPill({ icon, label, value }: StatusPillProps) {
  return (
    <div className="min-w-0 max-w-52 rounded-xl border border-line bg-panel/85 px-3 py-2 shadow-terminal">
      <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
        {icon}
        {label}
      </div>
      <p className="mt-1 max-w-80 truncate font-mono text-xs text-foreground">{value}</p>
    </div>
  )
}

function ReadOnlyPill() {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-green/30 bg-green/10 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-green">
      <ShieldCheck size={14} />
      Read only
    </div>
  )
}
