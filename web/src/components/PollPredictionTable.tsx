import type { ReactNode } from 'react'

import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow } from '@/lib/types'

export interface PollPredictionTableProps {
  readonly rows: readonly PollPredictionRow[]
  readonly emptyMessage: string
}

export function PollPredictionTable({ rows, emptyMessage }: PollPredictionTableProps) {
  const orderedRows = [...rows].sort((left, right) => Math.abs(right.edge) - Math.abs(left.edge))

  return (
    <div className="overflow-hidden rounded-2xl border border-line/70">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-panelStrong/90 text-[10px] uppercase tracking-[0.2em] text-muted">
          <tr>
            <Th>Market</Th>
            <Th>Phrase</Th>
            <Th>Model</Th>
            <Th>Market</Th>
            <Th>Residual</Th>
            <Th>Bid / Ask</Th>
            <Th>Side</Th>
            <Th>Edge</Th>
            <Th>Cost</Th>
            <Th>Vol</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line/50">
          {orderedRows.length === 0 ? (
            <tr>
                <td colSpan={10} className="px-4 py-10 text-center text-muted">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            orderedRows.map((row) => (
              <tr key={`${row.market_ticker}-${row.side}`} className="bg-background/25">
                <Td strong>{row.market_ticker}</Td>
                <Td>{row.target_phrase}</Td>
                <Td tone="text-cyan">{formatProbability(row.model_probability)}</Td>
                <Td>{formatProbability(row.market_probability)}</Td>
                <Td tone={row.residual_delta >= 0 ? 'text-green' : 'text-red'}>
                  {formatSigned(row.residual_delta)}
                </Td>
                <Td>
                  {formatProbability(row.yes_bid)} / {formatProbability(row.yes_ask)}
                </Td>
                <Td tone={sideTone(row.side)}>{row.side}</Td>
                <Td tone={row.edge > 0 ? 'text-green' : 'text-muted'}>{formatSigned(row.edge)}</Td>
                <Td>{formatProbability(row.cost)}</Td>
                <Td>{formatInteger(row.volume)}</Td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

interface CellProps {
  readonly children: ReactNode
  readonly tone?: string
  readonly strong?: boolean
}

function Th({ children }: CellProps) {
  return <th className="px-4 py-3 font-semibold">{children}</th>
}

function Td({ children, tone = 'text-foreground', strong = false }: CellProps) {
  return (
    <td className={`px-4 py-4 font-mono ${tone} ${strong ? 'font-semibold' : ''}`}>{children}</td>
  )
}

function sideTone(side: string): string {
  if (side === 'YES') return 'text-green'
  if (side === 'NO') return 'text-red'
  return 'text-muted'
}
