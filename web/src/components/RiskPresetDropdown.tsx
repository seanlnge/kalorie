import { Check, ChevronsUpDown } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { formatProbability } from '@/lib/format'
import type { RiskPreset } from '@/lib/types'
import { cn } from '@/lib/utils'

export interface RiskPresetDropdownProps {
  readonly presets: readonly RiskPreset[]
  readonly selectedPresetId: string | null
  readonly onSelect: (presetId: string) => void
  readonly compact?: boolean
}

export function RiskPresetDropdown({
  presets,
  selectedPresetId,
  onSelect,
  compact = false,
}: RiskPresetDropdownProps) {
  const [open, setOpen] = useState(false)
  const selected = presets.find((preset) => preset.id === selectedPresetId) ?? presets[0] ?? null

  return (
    <div className="min-w-0">
      <span className={compact ? 'sr-only' : 'mb-1 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted'}>
        Risk tolerance
      </span>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={[
              'w-full min-w-0 justify-between bg-background px-3 normal-case tracking-normal',
              compact ? 'h-9 min-w-[12rem]' : 'h-12',
            ].join(' ')}
          >
            <span className="truncate font-mono text-sm font-semibold">
              {selected?.label ?? 'Select risk preset'}
            </span>
            <ChevronsUpDown size={15} className="shrink-0 text-muted" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[min(32rem,calc(100vw-2rem))] p-0">
          <Command>
            <CommandInput placeholder="Search presets..." />
            <CommandList>
              <CommandEmpty>No preset found.</CommandEmpty>
              <CommandGroup heading="Risk presets">
                {presets.map((preset) => (
                  <CommandItem
                    key={preset.id}
                    value={`${preset.label} ${preset.id}`}
                    onSelect={() => {
                      onSelect(preset.id)
                      setOpen(false)
                    }}
                  >
                    <Check
                      size={14}
                      className={cn(
                        'text-cyan',
                        selected?.id === preset.id ? 'opacity-100' : 'opacity-0',
                      )}
                    />
                    <span className="min-w-0 flex-1 truncate">{preset.label}</span>
                    <span className="text-[10px] uppercase tracking-[0.12em] text-muted">
                      {preset.trade_side.replace('_', '-')}
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {compact ? null : (
        <div className="mt-2 grid grid-cols-4 overflow-hidden rounded-md border border-line bg-panel/80">
          <PresetCell label="Margin" value={formatProbability(selected?.min_margin)} />
          <PresetCell label="Kelly" value={formatProbability(selected?.kelly_fraction)} />
          <PresetCell label="Position" value={formatProbability(selected?.max_position_fraction)} />
          <PresetCell label="Event cap" value={formatProbability(selected?.max_event_exposure_fraction)} />
        </div>
      )}
    </div>
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
