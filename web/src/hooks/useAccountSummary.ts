import { useCallback, useEffect, useState } from 'react'

import { getAccountSummary } from '@/lib/api'
import type { AccountSummary } from '@/lib/types'

const ACCOUNT_POLL_INTERVAL_MS = 60_000

const PAPER_ACCOUNT: AccountSummary = {
  available: false,
  source: 'paper',
  portfolio_value: null,
  free_cash: null,
  position_exposure: null,
  bankroll: 100,
  error: null,
}

export function useAccountSummary() {
  const [summary, setSummary] = useState<AccountSummary>(PAPER_ACCOUNT)
  const [error, setError] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  const refresh = useCallback(() => setRefreshNonce((current) => current + 1), [])

  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      try {
        const nextSummary = await getAccountSummary()
        if (!cancelled) {
          setSummary(nextSummary)
          setError(nextSummary.error ?? null)
        }
      } catch (err) {
        if (!cancelled) {
          setSummary(PAPER_ACCOUNT)
          setError(err instanceof Error ? err.message : 'Failed to load account summary')
        }
      } finally {
        if (!cancelled) {
          timeoutId = setTimeout(poll, ACCOUNT_POLL_INTERVAL_MS)
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
