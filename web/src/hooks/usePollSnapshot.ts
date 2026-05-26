import { useCallback, useEffect, useState } from 'react'

import { getLatestPoll, getLatestTrades, getPollHistory } from '@/lib/api'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

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
    void refresh()
  }, [refresh])

  return { snapshot, trades, history, loading, error, refresh } as const
}
