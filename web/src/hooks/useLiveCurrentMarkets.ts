import { useCallback, useEffect, useRef, useState } from 'react'

import { currentMarketsStreamUrl, getCurrentMarkets } from '@/lib/api'
import type {
  CurrentMarketsStreamMessage,
  CurrentMarketsStreamStatus,
  PollPredictionRow,
  PollSnapshot,
  RiskPreset,
} from '@/lib/types'

const FULL_REFRESH_INTERVAL_MS = 60 * 60 * 1_000
const STREAM_RECONNECT_INTERVAL_MS = 5_000

export function useLiveCurrentMarkets(modelName: string | null, riskPreset: RiskPreset | null) {
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [streamStatus, setStreamStatus] = useState<CurrentMarketsStreamStatus>('idle')
  const [refreshNonce, setRefreshNonce] = useState(0)
  const snapshotRef = useRef<PollSnapshot | null>(null)
  const lastFetchRef = useRef<{ modelName: string; riskPresetId: string } | null>(null)

  const refresh = useCallback(() => {
    setRefreshNonce((current) => current + 1)
  }, [])

  useEffect(() => {
    if (!modelName || !riskPreset) {
      setSnapshot(null)
      setLoading(false)
      setError(null)
      return
    }
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null
    const previousFetch = lastFetchRef.current
    const riskOnlyChange =
      previousFetch?.modelName === modelName &&
      previousFetch.riskPresetId !== riskPreset.id &&
      snapshotRef.current !== null

    const refreshSnapshot = async ({
      refreshMarkets,
      forceModelRun,
    }: {
      refreshMarkets: boolean
      forceModelRun: boolean
    }) => {
      setLoading(true)
      setError(null)
      try {
        const nextSnapshot = await getCurrentMarkets(modelName, riskPreset, {
          refreshMarkets,
          forceModelRun,
        })
        if (!cancelled) {
          setSnapshot(nextSnapshot)
          snapshotRef.current = nextSnapshot
          lastFetchRef.current = { modelName, riskPresetId: riskPreset.id }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load current markets')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          timeoutId = setTimeout(() => {
            void refreshSnapshot({ refreshMarkets: true, forceModelRun: true })
          }, FULL_REFRESH_INTERVAL_MS)
        }
      }
    }

    if (!riskOnlyChange) {
      setSnapshot(null)
      snapshotRef.current = null
    }
    void refreshSnapshot({
      refreshMarkets: !riskOnlyChange,
      forceModelRun: refreshNonce > 0,
    })

    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [modelName, riskPreset, refreshNonce])

  useEffect(() => {
    if (!modelName || !riskPreset || !snapshot) {
      setStreamStatus('idle')
      return
    }
    let cancelled = false
    let websocket: WebSocket | null = null
    let reconnectId: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      setStreamStatus('connecting')
      websocket = new WebSocket(currentMarketsStreamUrl(modelName, riskPreset))
      websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(String(event.data)) as CurrentMarketsStreamMessage
          handleStreamMessage(message)
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to parse market stream update')
          setStreamStatus('error')
        }
      }
      websocket.onerror = () => {
        setStreamStatus('error')
      }
      websocket.onclose = () => {
        if (cancelled) return
        setStreamStatus('stale')
        reconnectId = setTimeout(connect, STREAM_RECONNECT_INTERVAL_MS)
      }
    }

    const handleStreamMessage = (message: CurrentMarketsStreamMessage) => {
      if (message.type === 'status') {
        if (message.status === 'subscribed') {
          setStreamStatus('live')
          return
        }
        setStreamStatus(message.status === 'error' ? 'error' : 'stale')
        if (message.message) {
          setError(message.message)
        }
        return
      }
      mergeRowUpdate(message.row)
    }

    const mergeRowUpdate = (row: PollPredictionRow) => {
      setSnapshot((current) => {
        if (!current) return current
        const predictionRows = current.prediction_rows.map((existing) =>
          existing.market_ticker === row.market_ticker ? row : existing,
        )
        const tradeRows = predictionRows.filter(isTradeRow)
        const nextSnapshot = {
          ...current,
          completed_at: new Date().toISOString(),
          prediction_count: predictionRows.length,
          trade_count: tradeRows.length,
          prediction_rows: predictionRows,
          trade_rows: tradeRows,
        }
        snapshotRef.current = nextSnapshot
        return nextSnapshot
      })
    }

    connect()
    return () => {
      cancelled = true
      if (reconnectId) {
        clearTimeout(reconnectId)
      }
      websocket?.close()
    }
  }, [modelName, riskPreset, snapshot?.poll_id])

  return { snapshot, loading, error, streamStatus, refresh } as const
}

function isTradeRow(row: PollPredictionRow): boolean {
  return row.side === 'YES' || row.side === 'NO'
}
