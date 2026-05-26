import { ArrowDown, ArrowUp, Minus } from 'lucide-react'
import type { ReactNode } from 'react'

import { formatProbability, formatSigned } from '@/lib/format'
import type { ScoreRow } from '@/lib/types'

export interface PredictionTableProps {
  readonly rows: readonly ScoreRow[]
}

export function PredictionTable({ rows }: PredictionTableProps) {
  const orderedRows = [...rows].sort((left, right) => Math.abs(right.edge) - Math.abs(left.edge))

  return (
    <section className="rounded-lg border border-line bg-panel/80 p-4 shadow-terminal">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted">
            Predictions
          </p>
          <h2 className="font-display text-lg font-semibold">Contract-level model output</h2>
        </div>
        <span className="rounded-full border border-line/70 bg-background/60 px-3 py-1 font-mono text-xs text-muted">
          {orderedRows.length} rows
        </span>
      </div>
      <div className="overflow-x-auto rounded-xl border border-line bg-background/45">
        <table className="w-full min-w-[920px] border-collapse text-left text-xs">
          <thead className="border-b border-line bg-panelStrong/90 font-mono text-[9px] uppercase tracking-[0.2em] text-muted">
            <tr>
              <Th>Market</Th>
              <Th>Event</Th>
              <Th>Model Prob</Th>
              <Th>Market Prob</Th>
              <Th>Residual</Th>
              <Th>Side</Th>
              <Th>Edge</Th>
              <Th>Cost</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/50">
            {orderedRows.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-10 text-center text-muted">
                  No prediction rows yet. Score a sample or upload a CSV.
                </td>
              </tr>
            ) : (
              orderedRows.map((row) => (
                <tr
                  key={`${row.market_ticker}-${row.event_ticker}`}
                  className="bg-background/20 transition hover:bg-panelStrong/45"
                >
                  <Td strong>{row.market_ticker}</Td>
                  <Td>{row.event_ticker}</Td>
                  <Td>{formatProbability(row.model_probability)}</Td>
                  <Td>{formatProbability(row.market_probability)}</Td>
                  <Td tone={row.residual_delta >= 0 ? 'text-green' : 'text-red'}>
                    {formatSigned(row.residual_delta)}
                  </Td>
                  <Td>
                    <SideBadge side={row.side} />
                  </Td>
                  <Td tone={row.edge > 0 ? 'text-green' : 'text-muted'}>{formatSigned(row.edge)}</Td>
                  <Td>{formatProbability(row.cost)}</Td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}

interface CellProps {
  readonly children: ReactNode
  readonly tone?: string
  readonly strong?: boolean
}

function Th({ children }: CellProps) {
  return <th className="px-3 py-3 font-semibold">{children}</th>
}

function Td({ children, tone = 'text-foreground', strong = false }: CellProps) {
  return (
    <td className={`px-3 py-3 font-mono ${tone} ${strong ? 'font-semibold' : ''}`}>{children}</td>
  )
}

interface SideBadgeProps {
  readonly side: string
}

function SideBadge({ side }: SideBadgeProps) {
  const config =
    side === 'YES'
      ? { icon: <ArrowUp size={13} />, className: 'border-green/30 bg-green/10 text-green' }
      : side === 'NO'
        ? { icon: <ArrowDown size={13} />, className: 'border-red/30 bg-red/10 text-red' }
        : { icon: <Minus size={13} />, className: 'border-line bg-background/60 text-muted' }

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] ${config.className}`}
    >
      {config.icon}
      {side}
    </span>
  )
}
