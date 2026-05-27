import { UploadCloud, Zap } from 'lucide-react'
import type { ChangeEvent } from 'react'
import { useRef, useState } from 'react'

import type { SampleRow } from '@/lib/types'

export interface ScoringPanelProps {
  readonly sampleRows: readonly SampleRow[]
  readonly selectedRowIndex: number
  readonly scoring: boolean
  readonly disabled?: boolean
  readonly onRowIndexChange: (rowIndex: number) => void
  readonly onScoreSample: () => Promise<void>
  readonly onScoreUpload: (file: File) => Promise<void>
}

export function ScoringPanel({
  sampleRows,
  selectedRowIndex,
  scoring,
  disabled = false,
  onRowIndexChange,
  onScoreSample,
  onScoreUpload,
}: ScoringPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    await onScoreUpload(file)
  }

  return (
    <section className="grid gap-4 lg:grid-cols-[1fr_1fr]">
      <div className="rounded-lg border border-line bg-panel/80 p-4 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">Scoring Panel</p>
        <h2 className="font-display text-lg font-semibold">Score a bundled training row</h2>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-2 text-sm text-muted">
            Sample row
            <select
              value={selectedRowIndex}
              disabled={disabled}
              onChange={(event) => onRowIndexChange(Number(event.target.value))}
              className="min-w-72 rounded-md border border-line bg-background px-3 py-3 font-mono text-sm text-foreground disabled:cursor-not-allowed disabled:opacity-50"
            >
              {sampleRows.map((row) => (
                <option key={row.row_index} value={row.row_index}>
                  #{row.row_index} {row.market_ticker ?? 'unknown market'}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void onScoreSample()}
            disabled={disabled || scoring || sampleRows.length === 0}
            className="inline-flex items-center gap-2 rounded-md bg-green px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-background transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Zap size={16} />
            {scoring ? 'Scoring...' : 'Score sample'}
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-dashed border-cyan/35 bg-cyan/5 p-4 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-cyan">CSV Upload</p>
        <h2 className="font-display text-lg font-semibold">Score an external row CSV</h2>
        <p className="mt-2 text-sm leading-6 text-muted">
          Use a CSV with the saved runtime column contract. The selected row index applies to the
          uploaded file.
        </p>
        <div className="mt-4 flex items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(event) => void handleFileChange(event)}
          />
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={disabled || scoring}
            className="inline-flex items-center gap-2 rounded-md border border-cyan/40 px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-cyan transition hover:bg-cyan/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <UploadCloud size={16} />
            Upload CSV
          </button>
          <p className="truncate font-mono text-xs text-muted">{fileName ?? 'No upload selected'}</p>
        </div>
      </div>
    </section>
  )
}
