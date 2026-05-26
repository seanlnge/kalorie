import { useCallback, useEffect, useState } from 'react'

import { getLatestPoll, getLatestTrades } from '@/lib/api'
import type { PollPredictionRow, PollSnapshot } from '@/lib/types'

export function usePollSnapshot() {
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [trades, setTrades] = useState<PollPredictionRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [nextSnapshot, nextTrades] = await Promise.all([getLatestPoll(), getLatestTrades()])
      setSnapshot(nextSnapshot)
      setTrades(nextTrades)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load poll cache')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { snapshot, trades, loading, error, refresh } as const
}
