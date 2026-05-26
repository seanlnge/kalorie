import { Boxes, CheckCircle2 } from 'lucide-react'

import { formatInteger } from '@/lib/format'
import type { SavedModelMetadata } from '@/lib/types'

export interface ModelSidebarProps {
  readonly models: readonly SavedModelMetadata[]
  readonly selectedModelName: string | null
  readonly onSelect: (modelName: string) => void
}

export function ModelSidebar({ models, selectedModelName, onSelect }: ModelSidebarProps) {
  return (
    <aside className="flex h-screen w-80 shrink-0 flex-col border-r border-line/60 bg-panel/80 backdrop-blur">
      <div className="border-b border-line/60 p-5">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl border border-cyan/30 bg-cyan/10 text-cyan">
            <Boxes size={20} />
          </div>
          <div>
            <p className="font-display text-lg font-semibold tracking-tight">Model Registry</p>
            <p className="text-xs uppercase tracking-[0.24em] text-muted">models/* bundles</p>
          </div>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {models.length === 0 ? (
          <div className="rounded-2xl border border-amber/30 bg-amber/10 p-4 text-sm text-amber">
            No valid saved models discovered.
          </div>
        ) : null}
        {models.map((model) => {
          const active = model.name === selectedModelName
          return (
            <button
              key={model.name}
              type="button"
              onClick={() => onSelect(model.name)}
              className={[
                'w-full rounded-2xl border p-4 text-left transition',
                active
                  ? 'border-cyan/60 bg-cyan/10 shadow-terminal'
                  : 'border-line/70 bg-panelStrong/50 hover:border-cyan/30 hover:bg-panelStrong',
              ].join(' ')}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-mono text-sm font-semibold text-foreground">{model.name}</p>
                  <p className="mt-1 text-xs text-muted">{model.model_type ?? 'saved model'}</p>
                </div>
                <span className="flex items-center gap-1 rounded-full border border-green/30 bg-green/10 px-2 py-1 text-[10px] uppercase tracking-wide text-green">
                  <CheckCircle2 size={12} />
                  {model.health}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                <RegistryStat label="Rows" value={formatInteger(model.training.row_count)} />
                <RegistryStat label="Events" value={formatInteger(model.training.event_count)} />
                <RegistryStat label="Features" value={formatInteger(model.training.feature_count)} />
              </div>
            </button>
          )
        })}
      </div>
    </aside>
  )
}

interface RegistryStatProps {
  readonly label: string
  readonly value: string
}

function RegistryStat({ label, value }: RegistryStatProps) {
  return (
    <div className="rounded-xl border border-line/50 bg-background/50 p-2">
      <p className="text-[10px] uppercase tracking-wide text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm text-foreground">{value}</p>
    </div>
  )
}
