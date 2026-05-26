import { Activity, Clock3 } from 'lucide-react'
import type { ReactNode } from 'react'

import type { SavedModelMetadata } from '@/lib/types'

export interface TopStatusBarProps {
  readonly selectedModel: SavedModelMetadata | null
}

export function TopStatusBar({ selectedModel }: TopStatusBarProps) {
  const timestamp = selectedModel?.trained_at
    ? new Date(selectedModel.trained_at).toLocaleString()
    : 'No model selected'

  return (
    <header className="sticky top-0 z-10 border-b border-line/60 bg-background/82 px-6 py-4 backdrop-blur-xl">
      <div className="flex items-center justify-between gap-5">
        <div>
          <p className="text-xs uppercase tracking-[0.36em] text-cyan">Kalorie2</p>
          <h1 className="font-display text-2xl font-semibold tracking-tight">
            Earnings Mention Prediction Workstation
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <StatusPill icon={<Activity size={14} />} label="Active model" value={selectedModel?.name ?? '--'} />
          <StatusPill icon={<Clock3 size={14} />} label="Trained" value={timestamp} />
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
    <div className="min-w-44 rounded-2xl border border-line/70 bg-panel/80 px-4 py-3">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted">
        {icon}
        {label}
      </div>
      <p className="mt-1 max-w-80 truncate font-mono text-xs text-foreground">{value}</p>
    </div>
  )
}
