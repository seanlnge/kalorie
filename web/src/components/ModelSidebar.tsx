import { Boxes, CheckCircle2, TerminalSquare } from 'lucide-react'

import { formatInteger } from '@/lib/format'
import type { SavedModelMetadata } from '@/lib/types'

export interface ModelSidebarProps {
  readonly models: readonly SavedModelMetadata[]
  readonly selectedModelName: string | null
  readonly onSelect: (modelName: string) => void
}

export function ModelSidebar({ models, selectedModelName, onSelect }: ModelSidebarProps) {
  return (
    <aside className="flex h-screen w-80 shrink-0 flex-col overflow-x-hidden border-r border-line bg-panel/92 backdrop-blur">
      <div className="border-b border-line p-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan/30 bg-cyan/10 text-cyan shadow-bloom">
            <Boxes size={20} />
          </div>
          <div>
            <p className="font-display text-xl font-bold tracking-tight">KALORIE</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">
              Model registry
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-x-hidden overflow-y-auto p-3">
        {models.length === 0 ? (
          <div className="rounded-xl border border-amber/30 bg-amber/10 p-4 text-sm text-amber">
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
                'w-full rounded-xl border p-3 text-left transition',
                active
                  ? 'border-cyan/70 bg-cyan/10 shadow-bloom'
                  : 'border-line bg-panelStrong/45 hover:border-cyan/30 hover:bg-panelStrong/75',
              ].join(' ')}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="break-all font-mono text-sm font-semibold text-foreground">
                    {model.name}
                  </p>
                  <p className="mt-1 text-xs text-muted">{model.model_type ?? 'saved model'}</p>
                </div>
                <span className="flex items-center gap-1 rounded-full border border-green/30 bg-green/10 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-green">
                  <CheckCircle2 size={12} />
                  {model.health}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                <RegistryStat label="Rows" value={formatInteger(model.training.row_count)} />
                <RegistryStat label="Events" value={formatInteger(model.training.event_count)} />
                <RegistryStat label="Features" value={formatInteger(model.training.feature_count)} />
              </div>
            </button>
          )
        })}
      </div>
      <div className="border-t border-line p-3">
        <div className="rounded-xl border border-line bg-background/70 p-3">
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
            <TerminalSquare size={13} />
            Poller
          </div>
          <p className="mt-2 font-mono text-[11px] leading-5 text-cyan">
            kalorie2-market-poller loop
          </p>
          <p className="mt-1 text-xs leading-5 text-muted">Read-only cache writer. Orders disabled.</p>
        </div>
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
    <div className="rounded-lg border border-line/60 bg-background/60 p-2">
      <p className="font-mono text-[9px] uppercase tracking-wide text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm text-foreground">{value}</p>
    </div>
  )
}
