import { formatCurrency, formatDollars, formatInteger, formatSigned } from '@/lib/format'
import type { OpenPosition, OpenPositionsSummary, PollPredictionRow } from '@/lib/types'

export function TradingHistoryPage({
  positions,
  currentMarketRows,
}: {
  readonly positions: OpenPositionsSummary
  readonly currentMarketRows: readonly PollPredictionRow[]
}) {
  const modelEv = totalModelEv(positions.positions, currentMarketRows)
  return (
    <section className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <HistoryMetric label="Execution" value="Not live" tone="text-amber" />
        <HistoryMetric label="Executed trades" value="--" />
        <HistoryMetric label="Open positions" value={formatInteger(positions.open_position_count)} />
        <HistoryMetric label="Realized PnL" value={formatCurrency(positions.realized_pnl)} />
      </div>

      <OpenPositionsPanel
        positions={positions}
        currentMarketRows={currentMarketRows}
        modelEv={modelEv}
      />

      <section className="rounded-lg border border-line bg-panel/82 p-5 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          Executed trade ledger
        </p>
        <h2 className="mt-2 font-display text-lg font-semibold">No trades have been placed yet</h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Poll snapshots and risk overlays are model tests, not executed trades. This ledger will stay
          empty until trading functionality records real fills, positions, and realized PnL.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <RoadmapStep label="1" title="Market scan" description="Current Markets ranks live opportunities." />
          <RoadmapStep label="2" title="Human review" description="Use model + risk preset output as research." />
          <RoadmapStep label="3" title="Execution later" description="Real fills will populate this ledger." />
        </div>
      </section>
    </section>
  )
}

function OpenPositionsPanel({
  positions,
  currentMarketRows,
  modelEv,
}: {
  readonly positions: OpenPositionsSummary
  readonly currentMarketRows: readonly PollPredictionRow[]
  readonly modelEv: number | null
}) {
  const rowByTicker = new Map(currentMarketRows.map((row) => [row.market_ticker, row]))
  return (
    <section className="rounded-lg border border-line bg-panel/82 p-5 shadow-terminal">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Open positions
          </p>
          <h2 className="mt-2 font-display text-lg font-semibold">
            {positions.available ? 'Live Kalshi exposure' : 'No authenticated positions'}
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            Current account positions are shown separately from the executed-trade ledger. Model EV
            appears when a position ticker is also in the active market scan.
          </p>
        </div>
      </div>
      <div className="grid gap-3 md:grid-cols-5">
        <HistoryMetric label="Contracts" value={formatInteger(positions.total_contracts)} />
        <HistoryMetric label="Avg buy" value={formatDollars(positions.average_price)} />
        <HistoryMetric label="Exposure" value={formatDollars(positions.total_exposure)} />
        <HistoryMetric label="Market value" value={formatDollars(positions.total_market_value)} />
        <HistoryMetric label="Model EV" value={formatCurrency(modelEv)} />
      </div>
      {positions.positions.length ? (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-background/70 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Market</th>
                <th className="px-3 py-2 text-left font-medium">Side</th>
                <th className="px-3 py-2 text-right font-medium">Contracts</th>
                <th className="px-3 py-2 text-right font-medium">Avg buy</th>
                <th className="px-3 py-2 text-right font-medium">Exposure</th>
                <th className="px-3 py-2 text-right font-medium">Value</th>
                <th className="px-3 py-2 text-right font-medium">EV/Contract</th>
                <th className="px-3 py-2 text-right font-medium">Position EV</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/70">
              {positions.positions.map((position) => {
                const modelRow = rowByTicker.get(position.market_ticker)
                const evPerContract = heldPositionModelEv(position, modelRow)
                const positionEv = evPerContract === null ? null : evPerContract * position.contracts
                return (
                  <tr key={position.market_ticker} className="hover:bg-panelStrong/35">
                    <td className="max-w-[20rem] break-all px-3 py-2 font-mono text-xs font-semibold text-foreground">
                      {position.market_ticker}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-muted">{position.side}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatInteger(position.contracts)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatDollars(position.average_price)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatDollars(position.exposure)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatDollars(position.market_value)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatSigned(evPerContract)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {formatCurrency(positionEv)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="mt-4 rounded-md border border-line bg-background/55 p-3 text-sm text-muted">
          No open Kalshi positions returned.
        </div>
      )}
    </section>
  )
}

function totalModelEv(
  positions: readonly OpenPosition[],
  currentMarketRows: readonly PollPredictionRow[],
): number | null {
  const rowByTicker = new Map(currentMarketRows.map((row) => [row.market_ticker, row]))
  let total = 0
  let matched = false
  for (const position of positions) {
    const modelRow = rowByTicker.get(position.market_ticker)
    const evPerContract = heldPositionModelEv(position, modelRow)
    if (evPerContract === null) continue
    total += evPerContract * position.contracts
    matched = true
  }
  return matched ? total : null
}

function heldPositionModelEv(
  position: OpenPosition,
  modelRow: PollPredictionRow | undefined,
): number | null {
  if (!modelRow || position.average_price === null || position.average_price === undefined) {
    return null
  }
  const winProbability =
    position.side === 'NO' ? 1 - modelRow.model_probability : modelRow.model_probability
  return winProbability - position.average_price
}

function HistoryMetric({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div className="rounded-lg border border-line bg-panel/75 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className={`mt-2 break-all font-mono text-lg font-semibold leading-6 ${tone}`}>{value}</p>
    </div>
  )
}

function RoadmapStep({
  label,
  title,
  description,
}: {
  readonly label: string
  readonly title: string
  readonly description: string
}) {
  return (
    <div className="rounded-md border border-line bg-background/55 p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan">Step {label}</p>
      <p className="mt-2 font-display text-sm font-semibold text-foreground">{title}</p>
      <p className="mt-1 text-xs leading-5 text-muted">{description}</p>
    </div>
  )
}
