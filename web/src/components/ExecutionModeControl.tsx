import type { ExecutionMode } from '@/lib/types'

export interface ExecutionModeControlProps {
  readonly mode: ExecutionMode
  readonly onChange: (mode: ExecutionMode) => void
}

export function ExecutionModeControl({ mode, onChange }: ExecutionModeControlProps) {
  return (
    <section className="rounded-3xl border border-line/70 bg-panel/70 p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-muted">Execution Controls</p>
          <h2 className="font-display text-lg font-semibold">Trade criteria</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
            YES when model probability is above pre-close YES ask. NO when model probability is
            below pre-close YES bid. Otherwise no trade.
          </p>
        </div>
        <div className="flex rounded-2xl border border-line/70 bg-background/60 p-1">
          <ModeButton active={mode === 'all'} label="All trades" onClick={() => onChange('all')} />
          <ModeButton active={mode === 'no_only'} label="NO-only" onClick={() => onChange('no_only')} />
        </div>
      </div>
    </section>
  )
}

interface ModeButtonProps {
  readonly active: boolean
  readonly label: string
  readonly onClick: () => void
}

function ModeButton({ active, label, onClick }: ModeButtonProps) {
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
