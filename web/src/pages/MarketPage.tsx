import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'

import { getEventMarkets, getRun, listRuns, submitJob } from '@/lib/api'
import type { MarketRow, RunInfo, RunResultPayload } from '@/lib/types'

import { Ex99Dropzone } from '@/components/markets/Ex99Dropzone'
import { PredictionTable } from '@/components/markets/PredictionTable'
import { RunSelector } from '@/components/markets/RunSelector'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

export function MarketPage() {
  const { ticker = '' } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const selectedRunFromQuery = searchParams.get('run')
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string | null>(selectedRunFromQuery)
  const [result, setResult] = useState<RunResultPayload | null>(null)
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [eventMarkets, setEventMarkets] = useState<MarketRow[]>([])
  const [loadingEventMarkets, setLoadingEventMarkets] = useState(true)

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id == selectedRunId) ?? null,
    [runs, selectedRunId]
  )

  const loadRun = useCallback(
    async (runId: string) => {
      const payload = await getRun(ticker, runId)
      setResult(payload.result)
      setSelectedRunId(payload.run.run_id)
      navigate(`/markets/${ticker}?run=${payload.run.run_id}`, { replace: true })
    },
    [navigate, ticker]
  )

  const refreshRuns = useCallback(async () => {
    if (!ticker) return
    setLoadingRuns(true)
    try {
      const nextRuns = await listRuns(ticker)
      setRuns(nextRuns)

      const preferredRunId =
        selectedRunFromQuery ??
        selectedRunId ??
        nextRuns.find((run) => run.status === 'completed')?.run_id ??
        nextRuns[0]?.run_id
      if (preferredRunId) {
        await loadRun(preferredRunId)
      } else {
        setResult(null)
      }
    } finally {
      setLoadingRuns(false)
    }
  }, [loadRun, selectedRunFromQuery, selectedRunId, ticker])

  useEffect(() => {
    const handle = window.setTimeout(() => {
      void refreshRuns()
    }, 0)
    return () => {
      window.clearTimeout(handle)
    }
  }, [refreshRuns])

  useEffect(() => {
    let cancelled = false
    const resolvedEventTicker = inferEventTicker(ticker)
    const handle = window.setTimeout(() => {
      setLoadingEventMarkets(true)
      void getEventMarkets(resolvedEventTicker)
        .then((markets) => {
          if (cancelled) {
            return
          }
          const rows = [...markets].sort((left, right) => {
            if (right.volume !== left.volume) {
              return right.volume - left.volume
            }
            return left.market_ticker.localeCompare(right.market_ticker)
          })
          setEventMarkets(rows)
        })
        .catch(() => {
          if (!cancelled) {
            setEventMarkets([])
          }
        })
        .finally(() => {
          if (!cancelled) {
            setLoadingEventMarkets(false)
          }
        })
    }, 0)

    return () => {
      cancelled = true
      window.clearTimeout(handle)
    }
  }, [ticker])

  async function handleSubmit(files: File[]) {
    if (!ticker) return
    const payload = await submitJob(ticker, { files })
    await refreshRuns()
    toast.success(`Run queued for ${ticker}`, {
      action: {
        label: 'Open Run',
        onClick: () => {
          void loadRun(payload.run.run_id)
        },
      },
    })
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">
            <Link to="/" className="text-primary hover:underline">
              Back to markets
            </Link>
          </p>
          <h1 className="text-xl font-semibold">{inferEventTicker(ticker)}</h1>
          <p className="text-xs text-muted-foreground">Anchor contract: {ticker}</p>
        </div>
        <RunSelector runs={runs} selectedRunId={selectedRunId} onSelect={(runId) => void loadRun(runId)} />
      </div>

      <Ex99Dropzone onSubmit={handleSubmit} />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Active Run</CardTitle>
        </CardHeader>
        <CardContent className="text-sm">
          {loadingRuns ? 'Loading runs...' : selectedRun ? selectedRun.run_id : 'No runs yet for this market.'}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Event Contracts (Live Orderbook Snapshot)</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Phrase</TableHead>
                <TableHead>Market</TableHead>
                <TableHead>Yes Bid</TableHead>
                <TableHead>Yes Ask</TableHead>
                <TableHead>Spread</TableHead>
                <TableHead>Volume</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loadingEventMarkets ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-muted-foreground">
                    Loading event contracts...
                  </TableCell>
                </TableRow>
              ) : eventMarkets.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-muted-foreground">
                    No open contracts found for this event.
                  </TableCell>
                </TableRow>
              ) : (
                eventMarkets.map((market) => (
                  <TableRow key={market.market_ticker}>
                    <TableCell>{market.target_phrase}</TableCell>
                    <TableCell className="max-w-[24rem] truncate">{market.market_ticker}</TableCell>
                    <TableCell>{market.yes_bid}</TableCell>
                    <TableCell>{market.yes_ask}</TableCell>
                    <TableCell>{market.spread}</TableCell>
                    <TableCell>{market.volume}</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <PredictionTable result={result} />
    </section>
  )
}

function inferEventTicker(marketTicker: string): string {
  const dashIndex = marketTicker.lastIndexOf('-')
  if (dashIndex <= 0) {
    return marketTicker
  }
  return marketTicker.slice(0, dashIndex)
}

