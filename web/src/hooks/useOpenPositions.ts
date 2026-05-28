import { useCallback, useEffect, useState } from 'react'

import { getOpenPositionsSummary } from '@/lib/api'
import type { OpenPositionsSummary } from '@/lib/types'

const POSITIONS_POLL_INTERVAL_MS = 60_000

const EMPTY_POSITIONS: OpenPositionsSummary = {
  available: false,
  source: 'paper',
  open_position_count: 0,
  total_contracts: 0,
  average_price: null,
  total_market_value: null,
  total_exposure: null,
  realized_pnl: null,
  fees_paid: null,
  positions: [],
  error: null,
}

export function useOpenPositions() {
  const [summary, setSummary] = useState<OpenPositionsSummary>(EMPTY_POSITIONS)
  const [error, setError] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  const refresh = useCallback(() => setRefreshNonce((current) => current + 1), [])

  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      try {
        const nextSummary = await getOpenPositionsSummary()
        if (!cancelled) {
          setSummary(nextSummary)
          setError(nextSummary.error ?? null)
        }
      } catch (err) {
        if (!cancelled) {
          setSummary(EMPTY_POSITIONS)
          setError(err instanceof Error ? err.message : 'Failed to load open positions')
        }
      } finally {
        if (!cancelled) {
          timeoutId = setTimeout(poll, POSITIONS_POLL_INTERVAL_MS)
        }
      }
    }

    void poll()
    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [refreshNonce])

  return { summary, error, refresh } as const
}
