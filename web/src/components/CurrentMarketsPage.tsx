import { ChevronDown, RefreshCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { formatDollars, formatInteger, formatProbability, formatSigned } from '@/lib/format'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

export interface CurrentMarketsPageProps {
  readonly snapshot: PollSnapshot | null
  readonly loading: boolean
  readonly bankroll: number
  readonly onRefresh: () => void
}

export function CurrentMarketsPage({ snapshot, loading, bankroll, onRefresh }: CurrentMarketsPageProps) {
  const eventGroups = useMemo(() => groupByEvent(snapshot?.prediction_rows ?? []), [snapshot])
  const [closedEvents, setClosedEvents] = useState<Set<string>>(new Set())
  const [now, setNow] = useState(() => Date.now())
  const initializedEventLayout = useRef(false)

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    setClosedEvents((current) => {
      if (eventGroups.length === 0) {
        initializedEventLayout.current = false
        return new Set<string>()
      }

      const eventTickers = new Set(eventGroups.map((group) => group.eventTicker))
      if (!initializedEventLayout.current) {
        initializedEventLayout.current = true
        const openTickers = new Set(
          eventGroups.filter((group) => group.tradeCount > 0).map((group) => group.eventTicker),
        )
        if (openTickers.size === 0) {
          openTickers.add(eventGroups[0].eventTicker)
        }
        return new Set(
          eventGroups
            .filter((group) => !openTickers.has(group.eventTicker))
            .map((group) => group.eventTicker),
        )
      }

      const next = new Set<string>()
      for (const eventTicker of current) {
        if (eventTickers.has(eventTicker)) {
          next.add(eventTicker)
        }
      }
      return next
    })
  }, [eventGroups])

  const expandAllEvents = () => setClosedEvents(new Set())
  const collapseAllEvents = () =>
    setClosedEvents(new Set(eventGroups.map((group) => group.eventTicker)))

  return (
    <section className="space-y-4">
      <div className="grid gap-3 md:grid-cols-[repeat(4,minmax(0,1fr))_auto]">
        <MarketMetric label="Poll" value={formatPollTimers(snapshot, now, loading)} />
        <MarketMetric label="Events" value={formatInteger(eventGroups.length)} />
        <MarketMetric label="Markets" value={formatInteger(snapshot?.market_count)} />
        <MarketMetric label="Trades" value={formatInteger(snapshot?.trade_count)} tone="text-green" />
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-line bg-background px-4 py-3 font-mono text-xs font-semibold uppercase tracking-[0.14em] text-foreground transition hover:border-cyan/60"
        >
          <RefreshCcw size={16} />
          Refresh
        </button>
      </div>

      {eventGroups.length > 0 ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-panel/70 px-4 py-3">
          <p className="max-w-2xl text-sm text-muted">
            Showing trade-bearing events open first. Expand the full board when you want to inspect
            every contract.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={expandAllEvents}
              className="rounded border border-line bg-background px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-cyan transition hover:border-cyan/60"
            >
              Expand all
            </button>
            <button
              type="button"
              onClick={collapseAllEvents}
              className="rounded border border-line bg-background px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted transition hover:border-muted/60 hover:text-foreground"
            >
              Collapse all
            </button>
          </div>
        </div>
      ) : null}

      <div className="space-y-3">
        {eventGroups.length === 0 ? (
          loading ? (
            <InferenceSkeletonCard />
          ) : (
            <div className="rounded-lg border border-amber/35 bg-amber/10 p-5 text-sm text-amber">
              No active markets returned yet.
            </div>
          )
        ) : (
          eventGroups.map((group) => (
            <EventMarketGroup
              key={group.eventTicker}
              group={group}
              bankroll={bankroll}
              open={!closedEvents.has(group.eventTicker)}
              onToggle={() =>
                setClosedEvents((current) => {
                  const next = new Set(current)
                  if (next.has(group.eventTicker)) {
                    next.delete(group.eventTicker)
                  } else {
                    next.add(group.eventTicker)
                  }
                  return next
                })
              }
            />
          ))
        )}
      </div>
    </section>
  )
}

interface EventGroup {
  eventTicker: string
  eventDatetime: string | null
  eventTitle: string
  totalExpectedEv: number
  tradeCount: number
  tradePercent: number
  evPer10Markets: number
  totalSpread: number
  averageSpread: number
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
      eventDatetime: eventRows.find((row) => row.event_datetime)?.event_datetime ?? null,
      eventTitle: eventRows.find((row) => row.event_title)?.event_title ?? '',
      totalExpectedEv: eventRows.reduce((total, row) => total + (row.ev_per_contract ?? row.edge), 0),
      tradeCount: eventRows.filter((row) => row.side === 'YES' || row.side === 'NO').length,
      tradePercent:
        eventRows.length > 0
          ? eventRows.filter((row) => row.side === 'YES' || row.side === 'NO').length /
            eventRows.length
          : 0,
      evPer10Markets:
        eventRows.length > 0
          ? (eventRows.reduce((total, row) => total + (row.ev_per_contract ?? row.edge), 0) /
              eventRows.length) *
            10
          : 0,
      totalSpread: eventRows.reduce((total, row) => total + (row.yes_ask - row.yes_bid), 0),
      averageSpread:
        eventRows.length > 0
          ? eventRows.reduce((total, row) => total + (row.yes_ask - row.yes_bid), 0) /
            eventRows.length
          : 0,
      rows: [...eventRows].sort((left, right) => right.volume - left.volume),
    }))
    .sort((left, right) => {
      const leftTime = eventSortTime(left.eventDatetime)
      const rightTime = eventSortTime(right.eventDatetime)
      if (leftTime !== rightTime) {
        return leftTime - rightTime
      }
      if (left.rows.length !== right.rows.length) {
        return right.rows.length - left.rows.length
      }
      return left.eventTicker.localeCompare(right.eventTicker)
    })
}

function EventMarketGroup({
  group,
  bankroll,
  open,
  onToggle,
}: {
  readonly group: EventGroup
  readonly bankroll: number
  readonly open: boolean
  readonly onToggle: () => void
}) {
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-panel/82 shadow-terminal">
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={onToggle}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onToggle()
          }
        }}
        className="flex w-full cursor-pointer flex-wrap items-center justify-between gap-3 border-b border-line bg-panelStrong/55 px-4 py-3 text-left outline-none transition hover:border-cyan/40 focus-visible:ring-1 focus-visible:ring-cyan/70"
      >
        <div className="min-w-0">
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">Event</p>
          <h2 className="font-display text-base font-semibold text-foreground">
            {group.eventTitle || group.eventTicker}
          </h2>
          <p className="mt-0.5 break-all font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
            {group.eventTicker}
          </p>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-cyan">
            {formatEventDateTime(group.eventDatetime)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-foreground">
            Total EV {formatSigned(group.totalExpectedEv)}
          </span>
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
            Trade {formatProbability(group.tradePercent)}
          </span>
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
            EV/10 mkts {formatSigned(group.evPer10Markets)}
          </span>
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
            Σ spread {formatProbability(group.totalSpread)}
          </span>
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
            Avg spread {formatProbability(group.averageSpread)}
          </span>
          <span className="rounded border border-line bg-background px-3 py-1 font-mono text-xs text-muted">
            {group.rows.length} markets
          </span>
          <ChevronDown
            size={15}
            className={`text-muted transition-transform ${open ? '' : '-rotate-90'}`}
          />
        </div>
      </div>
      {open ? <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-background/70 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Market</th>
              <th className="px-3 py-2 text-left font-medium">Phrase</th>
              <th className="px-3 py-2 text-left font-medium">Action</th>
              <th className="px-3 py-2 text-right font-medium">Expected</th>
              <th className="px-3 py-2 text-right font-medium">Bid</th>
              <th className="px-3 py-2 text-right font-medium">Ask</th>
              <th className="px-3 py-2 text-right font-medium">Spread</th>
              <th className="px-3 py-2 text-right font-medium">EV/Contract</th>
              <th className="px-3 py-2 text-right font-medium">Buy</th>
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
                <td className="px-3 py-2">
                  <SideBadge side={row.side} />
                </td>
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
                  {formatSigned(row.ev_per_contract ?? row.edge)}
                </td>
                <td className="px-3 py-2 text-right">
                  <TradeSize row={row} bankroll={bankroll} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div> : null}
    </section>
  )
}

function TradeSize({
  row,
  bankroll,
}: {
  readonly row: PollPredictionRow
  readonly bankroll: number
}) {
  const size = tradeSize(row, bankroll)
  if (!size) {
    return <span className="font-mono text-xs text-muted">--</span>
  }
  return (
    <div className="font-mono">
      <p className="text-sm font-semibold text-green">{formatDollars(size.dollars)}</p>
      <p className="text-[10px] uppercase tracking-[0.12em] text-muted">
        {formatInteger(size.contracts)} @ {formatDollars(size.contractCost)}
      </p>
    </div>
  )
}

function tradeSize(
  row: PollPredictionRow,
  bankroll: number,
): { dollars: number; contracts: number; contractCost: number } | null {
  if (row.side !== 'YES' && row.side !== 'NO') return null
  if (row.passes_risk_filter === false) return null
  const recommendedFraction = row.recommended_fraction ?? 0
  if (recommendedFraction <= 0 || bankroll <= 0 || row.cost <= 0) return null
  const rawDollars = bankroll * recommendedFraction
  const contracts = Math.floor(rawDollars / row.cost)
  if (contracts <= 0) return null
  return {
    dollars: Number((contracts * row.cost).toFixed(2)),
    contracts,
    contractCost: row.cost,
  }
}

function SideBadge({ side }: { readonly side: string }) {
  return (
    <span
      className={`inline-flex rounded-full border px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] ${sideTone(side)}`}
    >
      {side === 'NONE' ? 'No trade' : side}
    </span>
  )
}

function sideTone(side: string): string {
  if (side === 'YES') return 'border-green/30 bg-green/10 text-green'
  if (side === 'NO') return 'border-red/30 bg-red/10 text-red'
  return 'border-line bg-background/60 text-muted'
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

function formatPollTimers(snapshot: PollSnapshot | null, now: number, loading: boolean): string {
  const kal = loading && !snapshot ? 'polling...' : formatTimeLeft(snapshot?.next_market_poll_at, now)
  const run = formatTimeLeft(snapshot?.next_model_run_at, now)
  return `Kal: ${kal}; Run: ${run}`
}

function formatTimeLeft(value: string | null | undefined, now: number): string {
  if (!value) return '--'
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) return '--'
  const seconds = Math.max(0, Math.ceil((parsed - now) / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  if (minutes < 60) return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`
}

function eventSortTime(value: string | null): number {
  if (!value) {
    return Number.POSITIVE_INFINITY
  }
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? Number.POSITIVE_INFINITY : parsed
}

function formatEventDateTime(value: string | null): string {
  if (!value) {
    return 'Event date unknown on Kalshi'
  }
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) {
    return 'Event date unknown on Kalshi'
  }
  const date = new Date(parsed)
  if (
    date.getUTCHours() === 0 &&
    date.getUTCMinutes() === 0 &&
    date.getUTCSeconds() === 0
  ) {
    return `Kalshi event date ${date.toLocaleDateString(undefined, { timeZone: 'UTC' })}`
  }
  return `Kalshi event time ${date.toLocaleString()}`
}

function formatTimeSince(value: string | undefined, now: number, loading: boolean): string {
  if (!value) {
    return loading ? 'Polling now...' : '--'
  }
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) {
    return '--'
  }
  const elapsedMs = Math.max(0, now - parsed)
  if (elapsedMs < 5_000) {
    return 'just now'
  }
  const seconds = Math.floor(elapsedMs / 1_000)
  if (seconds < 60) {
    return `${seconds}s ago`
  }
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) {
    return `${minutes}m ago`
  }
  const hours = Math.floor(minutes / 60)
  if (hours < 24) {
    return `${hours}h ago`
  }
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function InferenceSkeletonCard() {
  return (
    <div className="rounded-lg border border-line bg-panel/82 p-5 text-sm text-muted shadow-terminal">
      <span className="inline-flex items-center gap-2 font-mono">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line" />
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line [animation-delay:160ms]" />
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-line [animation-delay:320ms]" />
        Running model inference for current active markets...
      </span>
    </div>
  )
}
