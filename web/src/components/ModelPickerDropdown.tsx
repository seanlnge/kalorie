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
import type { SavedModelMetadata } from '@/lib/types'
import { cn } from '@/lib/utils'

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
  const [open, setOpen] = useState(false)
  const selected = models.find((model) => model.name === selectedModelName) ?? models[0] ?? null
  const preview = selected?.model_card_preview

  return (
    <div className="min-w-0">
      <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Model
      </span>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="h-12 w-full justify-between rounded-md border-line bg-background px-3 text-left normal-case tracking-normal"
          >
            <span className="truncate font-mono text-sm font-semibold">
              {selected?.name ?? 'Select model'}
            </span>
            <ChevronsUpDown size={15} className="shrink-0 text-muted" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[min(32rem,calc(100vw-2rem))] p-0">
          <Command>
            <CommandInput placeholder="Search models..." />
            <CommandList>
              <CommandEmpty>No model found.</CommandEmpty>
              <CommandGroup heading="Saved models">
                {models.map((model) => (
                  <CommandItem
                    key={model.name}
                    value={model.name}
                    onSelect={() => {
                      onSelect(model.name)
                      setOpen(false)
                    }}
                  >
                    <Check
                      size={14}
                      className={cn(
                        'text-cyan',
                        selectedModelName === model.name ? 'opacity-100' : 'opacity-0',
                      )}
                    />
                    <span className="truncate">{model.name}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      <div className="mt-2 grid grid-cols-4 overflow-hidden rounded-md border border-line bg-panel/80">
        <PreviewCell label="Brier" value={preview?.brier?.toFixed(4) ?? '--'} />
        <PreviewCell label="Market" value={preview?.market_brier?.toFixed(4) ?? '--'} />
        <PreviewCell label="ECE" value={preview?.ece?.toFixed(4) ?? '--'} />
        <PreviewCell label="Log loss" value={preview?.log_loss?.toFixed(4) ?? '--'} />
      </div>
    </div>
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
