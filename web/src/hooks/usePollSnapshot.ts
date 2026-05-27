import { useCallback, useEffect, useState } from 'react'

import { getLatestPoll, getLatestTrades, getPollHistory } from '@/lib/api'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

const POLL_INTERVAL_MS = 60_000

export function usePollSnapshot() {
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [trades, setTrades] = useState<PollPredictionRow[]>([])
  const [history, setHistory] = useState<PollSnapshot[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [nextSnapshot, nextTrades, nextHistory] = await Promise.all([
        getLatestPoll(),
        getLatestTrades(),
        getPollHistory(),
      ])
      setSnapshot(nextSnapshot)
      setTrades(nextTrades)
      setHistory(nextHistory)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load poll cache')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      if (cancelled) {
        return
      }
      await refresh()
      if (!cancelled) {
        timeoutId = setTimeout(poll, POLL_INTERVAL_MS)
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [refresh])

  return { snapshot, trades, history, loading, error, refresh } as const
}
