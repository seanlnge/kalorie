import { ChevronDown } from 'lucide-react'

import { formatProbability, formatSigned } from '@/lib/format'
import type { SavedModelMetadata } from '@/lib/types'

export interface ModelPickerDropdownProps {
  readonly models: readonly SavedModelMetadata[]
  readonly selectedModelName: string | null
  readonly onSelect: (modelName: string) => void
}

export function ModelPickerDropdown({
  models,
  selectedModelName,
  onSelect,
}: ModelPickerDropdownProps) {
  const selected = models.find((model) => model.name === selectedModelName) ?? models[0] ?? null
  const preview = selected?.model_card_preview

  return (
    <label className="min-w-0">
      <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Model
      </span>
      <div className="grid gap-2 lg:grid-cols-[minmax(17rem,24rem)_auto]">
        <div className="relative">
          <select
            value={selectedModelName ?? ''}
            onChange={(event) => onSelect(event.target.value)}
            className="h-full w-full appearance-none rounded-md border border-line bg-background px-3 py-3 pr-9 font-mono text-sm font-semibold text-foreground shadow-terminal"
          >
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
          <ChevronDown
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted"
            size={16}
          />
        </div>
        <div className="grid grid-cols-3 overflow-hidden rounded-md border border-line bg-panel/80">
          <PreviewCell label="Trade %" value={formatProbability(preview?.trade_percent)} />
          <PreviewCell label="Brier" value={preview?.brier?.toFixed(4) ?? '--'} />
          <PreviewCell label="EV / 10" value={formatSigned(preview?.ev_per_10_trades)} tone="text-green" />
        </div>
      </div>
    </label>
  )
}

function PreviewCell({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div className="border-r border-line px-3 py-2 last:border-r-0">
      <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted">{label}</p>
      <p className={`mt-1 whitespace-nowrap font-mono text-sm font-semibold ${tone}`}>{value}</p>
    </div>
  )
}
