import type { RunResultPayload } from '@/lib/types'

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

interface PredictionTableProps {
  result: RunResultPayload | null
}

function rowValue(row: Record<string, unknown>, key: string): string {
  const value = row[key]
  return value === undefined || value === null ? '—' : String(value)
}

export function PredictionTable({ result }: PredictionTableProps) {
  const rows = result?.rows ?? []

  return (
    <div className="rounded-lg border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
        Predictions
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Phrase</TableHead>
            <TableHead>Yes Bid</TableHead>
            <TableHead>Yes Ask</TableHead>
            <TableHead>Spread</TableHead>
            <TableHead>Volume</TableHead>
            <TableHead>Model Prediction</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6} className="text-muted-foreground">
                No result rows yet for this run.
              </TableCell>
            </TableRow>
          ) : (
            rows.map((row, index) => {
              const currentRow = row as Record<string, unknown>
              return (
                <TableRow key={index}>
                  <TableCell>{rowValue(currentRow, 'target_phrase')}</TableCell>
                  <TableCell>{rowValue(currentRow, 'yes_bid')}</TableCell>
                  <TableCell>{rowValue(currentRow, 'yes_ask')}</TableCell>
                  <TableCell>{rowValue(currentRow, 'spread')}</TableCell>
                  <TableCell>{rowValue(currentRow, 'volume')}</TableCell>
                  <TableCell>{rowValue(currentRow, 'model_prediction')}</TableCell>
                </TableRow>
              )
            })
          )}
        </TableBody>
        <TableCaption>Contract-level prediction and market snapshot table.</TableCaption>
      </Table>
    </div>
  )
}

