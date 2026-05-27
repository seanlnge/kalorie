import { useCallback, useEffect, useState } from 'react'

import { getCurrentMarkets } from '@/lib/api'
import type { PollSnapshot, RiskPreset } from '@/lib/types'

const POLL_INTERVAL_MS = 60_000

export function useLiveCurrentMarkets(modelName: string | null, riskPreset: RiskPreset | null) {
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

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

    const poll = async () => {
      setLoading(true)
      setError(null)
      try {
        const nextSnapshot = await getCurrentMarkets(modelName, riskPreset)
        if (!cancelled) {
          setSnapshot(nextSnapshot)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load current markets')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          timeoutId = setTimeout(poll, POLL_INTERVAL_MS)
        }
      }
    }

    setSnapshot(null)
    void poll()

    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [modelName, riskPreset, refreshNonce])

  return { snapshot, loading, error, refresh } as const
}
