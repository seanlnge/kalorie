import { useCallback, useEffect, useRef, useState } from 'react'

import { getCurrentMarkets } from '@/lib/api'
import type { PollSnapshot, RiskPreset } from '@/lib/types'

const POLL_INTERVAL_MS = 60_000

export function useLiveCurrentMarkets(modelName: string | null, riskPreset: RiskPreset | null) {
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
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

    const poll = async (refreshMarkets: boolean) => {
      setLoading(true)
      setError(null)
      try {
        const nextSnapshot = await getCurrentMarkets(modelName, riskPreset, { refreshMarkets })
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
            void poll(true)
          }, POLL_INTERVAL_MS)
        }
      }
    }

    if (!riskOnlyChange) {
      setSnapshot(null)
      snapshotRef.current = null
    }
    void poll(!riskOnlyChange)

    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [modelName, riskPreset, refreshNonce])

  return { snapshot, loading, error, refresh } as const
}
