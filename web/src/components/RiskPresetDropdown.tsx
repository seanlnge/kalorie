import { ChevronDown } from 'lucide-react'

import { formatProbability } from '@/lib/format'
import type { RiskPreset } from '@/lib/types'

export interface RiskPresetDropdownProps {
  readonly presets: readonly RiskPreset[]
  readonly selectedPresetId: string | null
  readonly onSelect: (presetId: string) => void
}

export function RiskPresetDropdown({
  presets,
  selectedPresetId,
  onSelect,
}: RiskPresetDropdownProps) {
  const selected = presets.find((preset) => preset.id === selectedPresetId) ?? presets[0] ?? null

  return (
    <label className="min-w-0">
      <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Risk tolerance
      </span>
      <div className="grid gap-2 lg:grid-cols-[minmax(15rem,19rem)_auto]">
        <div className="relative">
          <select
            value={selectedPresetId ?? ''}
            onChange={(event) => onSelect(event.target.value)}
            className="h-full w-full appearance-none rounded-md border border-line bg-background px-3 py-3 pr-9 font-mono text-sm font-semibold text-foreground shadow-terminal"
          >
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label}
              </option>
            ))}
          </select>
          <ChevronDown
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted"
            size={16}
          />
        </div>
        <div className="grid grid-cols-4 overflow-hidden rounded-md border border-line bg-panel/80">
          <PresetCell label="Margin" value={formatProbability(selected?.min_margin)} />
          <PresetCell label="Kelly" value={formatProbability(selected?.kelly_fraction)} />
          <PresetCell label="Event cap" value={formatProbability(selected?.max_event_exposure_fraction)} />
          <PresetCell label="Ruin" value={selected?.risk_of_ruin_label ?? '--'} />
        </div>
      </div>
    </label>
  )
}

function PresetCell({
  label,
  value,
}: {
  readonly label: string
  readonly value: string
}) {
  return (
    <div className="border-r border-line px-3 py-2 last:border-r-0">
      <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted">{label}</p>
      <p className="mt-1 whitespace-nowrap font-mono text-sm font-semibold text-foreground">{value}</p>
    </div>
  )
}
