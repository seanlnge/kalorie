import { RefreshCcw, RadioTower } from 'lucide-react'
import { useMemo } from 'react'

import { formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

export interface CurrentMarketsPageProps {
  readonly snapshot: PollSnapshot | null
  readonly loading: boolean
  readonly onRefresh: () => void
}

export function CurrentMarketsPage({ snapshot, loading, onRefresh }: CurrentMarketsPageProps) {
  const eventGroups = useMemo(() => groupByEvent(snapshot?.prediction_rows ?? []), [snapshot])

  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-line bg-panelStrong/80 p-5 shadow-terminal">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.28em] text-cyan">
              <RadioTower size={14} />
              Current Markets
            </p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight">
              Active Kalshi market scan
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
              Grouped by event from the latest cached poll. Run `kalorie2-market-poller loop` to
              keep this view refreshed roughly every 10 minutes.
            </p>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-2 rounded-md border border-line bg-background px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-foreground transition hover:border-cyan/60"
          >
            <RefreshCcw size={16} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <MarketMetric label="Poll" value={snapshot?.poll_id ?? '--'} />
        <MarketMetric label="Events" value={formatInteger(eventGroups.length)} />
        <MarketMetric label="Markets" value={formatInteger(snapshot?.market_count)} />
        <MarketMetric label="Trades" value={formatInteger(snapshot?.trade_count)} tone="text-green" />
      </div>

      <div className="space-y-3">
        {eventGroups.length === 0 ? (
          <div className="rounded-lg border border-amber/35 bg-amber/10 p-5 text-sm text-amber">
            {loading ? 'Loading cached market scan...' : 'No poll cache yet. Run the market poller once.'}
          </div>
        ) : (
          eventGroups.map((group) => <EventMarketGroup key={group.eventTicker} group={group} />)
        )}
      </div>
    </section>
  )
}

interface EventGroup {
  eventTicker: string
  rows: PollPredictionRow[]
}

function groupByEvent(rows: PollPredictionRow[]): EventGroup[] {
  const groups = new Map<string, PollPredictionRow[]>()
  for (const row of rows) {
    const eventRows = groups.get(row.event_ticker) ?? []
    eventRows.push(row)
    groups.set(row.event_ticker, eventRows)
  }
  return [...groups.entries()]
    .map(([eventTicker, eventRows]) => ({
      eventTicker,
      rows: [...eventRows].sort((left, right) => right.volume - left.volume),
    }))
    .sort((left, right) => right.rows.length - left.rows.length)
}

function EventMarketGroup({ group }: { readonly group: EventGroup }) {
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-panel/82 shadow-terminal">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-panelStrong/55 px-4 py-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">Event</p>
          <h2 className="break-all font-mono text-sm font-semibold text-foreground">
            {group.eventTicker}
          </h2>
        </div>
        <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
          {group.rows.length} markets
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-background/70 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Market</th>
              <th className="px-3 py-2 text-left font-medium">Phrase</th>
              <th className="px-3 py-2 text-right font-medium">Expected</th>
              <th className="px-3 py-2 text-right font-medium">Bid</th>
              <th className="px-3 py-2 text-right font-medium">Ask</th>
              <th className="px-3 py-2 text-right font-medium">Spread</th>
              <th className="px-3 py-2 text-right font-medium">EV/Contract</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/70">
            {group.rows.map((row) => (
              <tr key={row.market_ticker} className="hover:bg-panelStrong/35">
                <td className="max-w-[18rem] px-3 py-2">
                  <p className="break-all font-mono text-xs font-semibold text-foreground">
                    {row.market_ticker}
                  </p>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
                    Vol {formatInteger(row.volume)}
                  </p>
                </td>
                <td className="px-3 py-2 text-muted">{row.target_phrase || '--'}</td>
                <td className="px-3 py-2 text-right font-mono text-foreground">
                  {formatProbability(row.model_probability)}
                </td>
                <td className="px-3 py-2 text-right font-mono">{formatProbability(row.yes_bid)}</td>
                <td className="px-3 py-2 text-right font-mono">{formatProbability(row.yes_ask)}</td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {formatProbability(row.yes_ask - row.yes_bid)}
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono font-semibold ${
                    row.edge >= 0 ? 'text-green' : 'text-red'
                  }`}
                >
                  {formatSigned(row.edge)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function MarketMetric({
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
