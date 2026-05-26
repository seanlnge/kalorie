import type { RunInfo } from '@/lib/types'

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface RunSelectorProps {
  runs: RunInfo[]
  selectedRunId: string | null
  onSelect: (runId: string) => void
}

export function RunSelector({ runs, selectedRunId, onSelect }: RunSelectorProps) {
  if (runs.length <= 1) {
    return null
  }

  return (
    <Select
      value={selectedRunId ?? undefined}
      onValueChange={(value) => {
        if (value) onSelect(value)
      }}
    >
      <SelectTrigger className="min-w-64">
        <SelectValue placeholder="Select run" />
      </SelectTrigger>
      <SelectContent>
        {runs.map((run) => (
          <SelectItem key={run.run_id} value={run.run_id}>
            {run.run_id} · {run.status}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

