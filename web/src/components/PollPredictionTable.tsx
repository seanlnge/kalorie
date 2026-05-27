import type { ReactNode } from 'react'

import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow } from '@/lib/types'

export interface PollPredictionTableProps {
  readonly rows: readonly PollPredictionRow[]
  readonly emptyMessage: string
  readonly loading?: boolean
}

export function PollPredictionTable({ rows, emptyMessage, loading = false }: PollPredictionTableProps) {
  const orderedRows = [...rows].sort((left, right) => Math.abs(right.edge) - Math.abs(left.edge))

  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-background/45">
      <table className="w-full min-w-[1120px] border-collapse text-left text-xs">
        <thead className="border-b border-line bg-panelStrong/90 font-mono text-[9px] uppercase tracking-[0.2em] text-muted">
          <tr>
            <Th>Market</Th>
            <Th>Phrase</Th>
            <Th numeric>Model</Th>
            <Th numeric>Market</Th>
            <Th numeric>Residual</Th>
            <Th numeric>Bid / Ask</Th>
            <Th>Side</Th>
            <Th numeric>Edge</Th>
            <Th numeric>Cost</Th>
            <Th numeric>Vol</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line/50">
          {orderedRows.length === 0 ? (
            <tr>
              <td colSpan={10} className="px-4 py-10 text-center text-muted">
                {loading ? <SkeletonDots /> : emptyMessage}
              </td>
            </tr>
          ) : (
            orderedRows.map((row) => (
              <tr
                key={`${row.market_ticker}-${row.side}`}
                className="bg-background/20 transition hover:bg-panelStrong/45"
              >
                <Td strong>{row.market_ticker}</Td>
                <Td>{row.target_phrase}</Td>
                <Td tone="text-cyan" numeric>
                  {formatProbability(row.model_probability)}
                </Td>
                <Td numeric>{formatProbability(row.market_probability)}</Td>
                <Td tone={row.residual_delta >= 0 ? 'text-green' : 'text-red'}>
                  {formatSigned(row.residual_delta)}
                </Td>
                <Td numeric>
                  {formatProbability(row.yes_bid)} / {formatProbability(row.yes_ask)}
                </Td>
                <Td>
                  <span
                    className={`rounded-full border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] ${sideTone(row.side)}`}
                  >
                    {row.side}
                  </span>
                </Td>
                <Td tone={row.edge > 0 ? 'text-green' : 'text-muted'} numeric>
                  {formatSigned(row.edge)}
                </Td>
                <Td numeric>{formatProbability(row.cost)}</Td>
                <Td numeric>{formatInteger(row.volume)}</Td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

function SkeletonDots() {
  return (
    <span className="inline-flex items-center gap-2 font-mono text-sm text-muted">
      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line" />
      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line [animation-delay:160ms]" />
      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line [animation-delay:320ms]" />
      Running model inference...
    </span>
  )
}

interface CellProps {
  readonly children: ReactNode
  readonly tone?: string
  readonly strong?: boolean
  readonly numeric?: boolean
}

function Th({ children, numeric = false }: CellProps) {
  return <th className={`px-3 py-3 font-semibold ${numeric ? 'text-right' : ''}`}>{children}</th>
}

function Td({ children, tone = 'text-foreground', strong = false, numeric = false }: CellProps) {
  return (
    <td
      className={`px-3 py-3 font-mono ${numeric ? 'text-right' : ''} ${tone} ${
        strong ? 'font-semibold' : ''
      }`}
    >
      {children}
    </td>
  )
}

function sideTone(side: string): string {
  if (side === 'YES') return 'border-green/30 bg-green/10 text-green'
  if (side === 'NO') return 'border-red/30 bg-red/10 text-red'
  return 'border-line bg-panel text-muted'
}
