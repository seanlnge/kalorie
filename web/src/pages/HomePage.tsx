import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { getOpenMarketsPayload } from '@/lib/api'
import type { MarketEventRow, MarketRow } from '@/lib/types'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

export function HomePage() {
  const [events, setEvents] = useState<MarketEventRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getOpenMarketsPayload()
      .then((payload) => {
        if (payload.events.length > 0) {
          setEvents(payload.events)
          return
        }
        setEvents(deriveEventsFromMarkets(payload.markets))
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to load markets')
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  return (
    <section className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Open Earnings Mention Markets</h1>
        <p className="text-sm text-muted-foreground">
          Select an earnings event to view all mention contracts and launch a model run.
        </p>
      </header>

      {loading ? <p>Loading markets...</p> : null}
      {error ? <p className="text-destructive">{error}</p> : null}

      {!loading && !error ? (
        <div className="rounded-lg border bg-card p-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Event</TableHead>
                <TableHead>Company</TableHead>
                <TableHead>Contracts</TableHead>
                <TableHead>Total Volume</TableHead>
                <TableHead>Example Phrase</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.map((event) => (
                <TableRow key={event.event_ticker}>
                  <TableCell>
                    <Link
                      to={`/markets/${event.representative_market_ticker}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {event.event_ticker}
                    </Link>
                  </TableCell>
                  <TableCell>{event.company_symbol}</TableCell>
                  <TableCell>{event.market_count}</TableCell>
                  <TableCell>{event.total_volume}</TableCell>
                  <TableCell className="max-w-[28rem] truncate">{event.representative_phrase}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : null}
    </section>
  )
}

function deriveEventsFromMarkets(markets: MarketRow[]): MarketEventRow[] {
  const grouped = new Map<string, MarketRow[]>()
  for (const market of markets) {
    const nextMarkets = grouped.get(market.event_ticker) ?? []
    nextMarkets.push(market)
    grouped.set(market.event_ticker, nextMarkets)
  }

  return Array.from(grouped.entries())
    .map(([eventTicker, eventMarkets]) => {
      const ordered = [...eventMarkets].sort((a, b) => {
        if (b.volume !== a.volume) {
          return b.volume - a.volume
        }
        return a.market_ticker.localeCompare(b.market_ticker)
      })
      const representative = ordered[0]
      return {
        event_ticker: eventTicker,
        company_symbol: representative.company_symbol,
        market_count: eventMarkets.length,
        total_volume: eventMarkets.reduce((sum, market) => sum + market.volume, 0),
        representative_market_ticker: representative.market_ticker,
        representative_phrase: representative.target_phrase,
      }
    })
    .sort((a, b) => a.event_ticker.localeCompare(b.event_ticker))
}

