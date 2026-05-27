import { Check, ChevronsUpDown, HelpCircle, Plus, Trash2 } from 'lucide-react'
import { type ReactNode, useMemo, useState } from 'react'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { formatProbability } from '@/lib/format'
import type { RiskPreset } from '@/lib/types'
import { cn } from '@/lib/utils'

export interface RiskPresetDropdownProps {
  readonly presets: readonly RiskPreset[]
  readonly selectedPresetId: string | null
  readonly onSelect: (presetId: string) => void
  readonly onCreate: (preset: RiskPreset) => void
  readonly onDelete: (presetId: string) => void
}

export function RiskPresetDropdown({
  presets,
  selectedPresetId,
  onSelect,
  onCreate,
  onDelete,
}: RiskPresetDropdownProps) {
  const [open, setOpen] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<RiskPreset | null>(null)
  const selected = presets.find((preset) => preset.id === selectedPresetId) ?? presets[0] ?? null
  const [draft, setDraft] = useState<RiskPreset>(() => makeDraftPreset(selected))
  const selectedCanDelete = Boolean(selected && presets.length > 1)

  const openCreateDialog = () => {
    setDraft(makeDraftPreset(selected))
    setDialogOpen(true)
  }

  const updateDraft = <K extends keyof RiskPreset>(key: K, value: RiskPreset[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
  }

  const submitDraft = () => {
    const normalized = {
      ...draft,
      id: uniquePresetId(draft.label || 'custom-preset', presets),
      label: draft.label.trim() || 'Custom preset',
    }
    onCreate(normalized)
    setDialogOpen(false)
  }

  return (
    <div className="min-w-0">
      <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
        Risk tolerance
      </span>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="h-12 w-full min-w-0 justify-between bg-background px-3 normal-case tracking-normal"
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
              <CommandGroup heading="Manage presets">
                <CommandItem
                  value="create new custom risk preset"
                  onSelect={() => {
                    setOpen(false)
                    openCreateDialog()
                  }}
                >
                  <Plus size={14} className="text-cyan" />
                  <span>Create preset from current selection</span>
                </CommandItem>
                <CommandItem
                  value="delete selected risk preset"
                  disabled={!selectedCanDelete || !selected}
                  onSelect={() => {
                    if (!selectedCanDelete || !selected) return
                    setOpen(false)
                    setDeleteTarget(selected)
                  }}
                >
                  <Trash2 size={14} className="text-red" />
                  <span>Delete selected preset</span>
                </CommandItem>
              </CommandGroup>
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
      <div className="mt-2 grid grid-cols-4 overflow-hidden rounded-md border border-line bg-panel/80">
        <PresetCell label="Margin" value={formatProbability(selected?.min_margin)} />
        <PresetCell label="Kelly" value={formatProbability(selected?.kelly_fraction)} />
        <PresetCell label="Position" value={formatProbability(selected?.max_position_fraction)} />
        <PresetCell label="Event cap" value={formatProbability(selected?.max_event_exposure_fraction)} />
      </div>
      <RiskPresetDialog
        open={dialogOpen}
        draft={draft}
        onOpenChange={setDialogOpen}
        onDraftChange={updateDraft}
        onSubmit={submitDraft}
      />
      <ConfirmDeleteDialog
        preset={deleteTarget}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => {
          if (!deleteTarget) return
          onDelete(deleteTarget.id)
          setDeleteTarget(null)
        }}
      />
    </div>
  )
}

interface RiskPresetDialogProps {
  readonly open: boolean
  readonly draft: RiskPreset
  readonly onOpenChange: (open: boolean) => void
  readonly onDraftChange: <K extends keyof RiskPreset>(key: K, value: RiskPreset[K]) => void
  readonly onSubmit: () => void
}

function RiskPresetDialog({
  open,
  draft,
  onOpenChange,
  onDraftChange,
  onSubmit,
}: RiskPresetDialogProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const fields = useMemo(
    () => [
      [
        'kelly_fraction',
        'Kelly %',
        draft.kelly_fraction,
        'Fraction of full Kelly sizing to use after the model clears the margin hurdle.',
      ] as const,
      [
        'max_position_fraction',
        'Position cap %',
        draft.max_position_fraction,
        'Maximum bankroll fraction allowed on any single market.',
      ] as const,
      [
        'max_event_exposure_fraction',
        'Event cap %',
        draft.max_event_exposure_fraction,
        'Maximum bankroll fraction allowed across all markets in one event.',
      ] as const,
    ],
    [draft],
  )

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Build risk preset</AlertDialogTitle>
          <AlertDialogDescription>
            Enter the policy fields once. Risk-of-ruin is computed later from the selected
            model and preset trial, not stored on the preset itself. Custom presets are kept in
            this browser session for live scoring; persist them later if you want them shared
            across reloads.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="grid gap-4 md:grid-cols-2">
          <Field
            label="Preset name"
            help="Human-readable name shown in the nav combobox."
          >
            <Input value={draft.label} onChange={(event) => onDraftChange('label', event.target.value)} />
          </Field>
          <Field
            label="Min margin %"
            help="Minimum edge over the market price before a trade can pass the risk filter."
          >
            <Input
              type="number"
              min="0"
              step="0.1"
              value={toPercentInput(draft.min_margin)}
              onChange={(event) => onDraftChange('min_margin', fromPercentInput(event.target.value))}
            />
          </Field>
          <div className="md:col-span-2">
            <Button
              type="button"
              variant="ghost"
              className="h-8 px-0 font-mono text-xs text-cyan"
              onClick={() => setShowAdvanced((current) => !current)}
            >
              {showAdvanced ? 'Hide advanced sizing fields' : 'Show advanced sizing fields'}
            </Button>
          </div>
          {showAdvanced ? (
            <>
              <Field
                label="Trade side"
                help="Restrict the strategy to YES, NO, or both sides when the predictive edge clears margin."
              >
                <select
                  value={draft.trade_side}
                  onChange={(event) =>
                    onDraftChange('trade_side', event.target.value as RiskPreset['trade_side'])
                  }
                  className="h-10 w-full rounded-md border border-line bg-background px-3 font-mono text-sm"
                >
                  <option value="all">All sides</option>
                  <option value="no_only">NO only</option>
                  <option value="yes_only">YES only</option>
                </select>
              </Field>
              {fields.map(([key, label, value, help]) => (
                <Field key={key} label={label} help={help}>
                  <Input
                    type="number"
                    min="0"
                    step="0.1"
                    value={toPercentInput(value)}
                    onChange={(event) => onDraftChange(key, fromPercentInput(event.target.value))}
                  />
                </Field>
              ))}
              <Field
                label="Description"
                help="Short note explaining when you would choose this risk profile."
              >
                <Input
                  value={draft.description}
                  onChange={(event) => onDraftChange('description', event.target.value)}
                />
              </Field>
            </>
          ) : null}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onSubmit}>Save preset</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function ConfirmDeleteDialog({
  preset,
  onCancel,
  onConfirm,
}: {
  readonly preset: RiskPreset | null
  readonly onCancel: () => void
  readonly onConfirm: () => void
}) {
  return (
    <AlertDialog open={Boolean(preset)} onOpenChange={(open) => {
      if (!open) onCancel()
    }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete risk preset?</AlertDialogTitle>
          <AlertDialogDescription>
            This removes {preset?.label ?? 'the selected preset'} from the current workstation
            session. It does not delete model artifacts or historical trial results.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={onCancel}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className="border-red/40 bg-red/15 text-red hover:bg-red/20"
          >
            Delete preset
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function Field({
  label,
  help,
  children,
}: {
  readonly label: string
  readonly help?: string
  readonly children: ReactNode
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label>{label}</Label>
        {help ? (
          <span
            title={help}
            className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-line text-muted"
          >
            <HelpCircle size={11} />
          </span>
        ) : null}
      </div>
      {children}
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

function makeDraftPreset(base: RiskPreset | null): RiskPreset {
  const source = base ?? {
    id: 'custom',
    label: 'Custom',
    description: 'Custom risk preset',
    trade_side: 'no_only',
    min_margin: 0.02,
    kelly_fraction: 0.5,
    max_position_fraction: 0.05,
    max_event_exposure_fraction: 0.12,
  }
  return {
    ...source,
    id: `${source.id}-copy`,
    label: `${source.label} Copy`,
    description: source.description || 'Custom risk preset',
  }
}

function uniquePresetId(label: string, presets: readonly RiskPreset[]): string {
  const slug = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'custom-preset'
  const existing = new Set(presets.map((preset) => preset.id))
  let candidate = slug
  let suffix = 2
  while (existing.has(candidate)) {
    candidate = `${slug}-${suffix}`
    suffix += 1
  }
  return candidate
}

function toPercentInput(value: number): string {
  return Number.isFinite(value) ? String(Number((value * 100).toFixed(3))) : '0'
}

function fromPercentInput(value: string): number {
  return clamp(Number(value || 0) / 100, 0, 1)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min))
}
