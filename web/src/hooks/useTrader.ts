import { useCallback, useEffect, useRef, useState } from 'react'

import {
  engageKillSwitch,
  getTraderActivity,
  getTraderStatus,
  restartTrader,
  resumeKillSwitch,
  startTrader,
  stopTrader,
} from '@/lib/api'
import type { RiskPreset, TraderActivityItem, TraderStatus } from '@/lib/types'

const TRADER_POLL_INTERVAL_MS = 5_000

interface UseTraderArgs {
  readonly stagedModelName: string | null
  readonly stagedRiskPreset: RiskPreset | null
}

export function useTrader({ stagedModelName, stagedRiskPreset }: UseTraderArgs) {
  const [status, setStatus] = useState<TraderStatus | null>(null)
  const [activity, setActivity] = useState<TraderActivityItem[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const pollNonce = useRef(0)

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextActivity] = await Promise.all([
        getTraderStatus(),
        getTraderActivity(200),
      ])
      setStatus(nextStatus)
      setActivity(nextActivity)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load trader status')
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      if (!cancelled) await refresh()
      if (!cancelled) timeoutId = setTimeout(poll, TRADER_POLL_INTERVAL_MS)
    }

    void poll()
    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [refresh, pollNonce])

  const runControl = useCallback(
    async (action: () => Promise<TraderStatus>) => {
      setBusy(true)
      try {
        const nextStatus = await action()
        setStatus(nextStatus)
        setError(null)
        await refresh()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Trader control failed')
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  const start = useCallback(() => {
    if (!stagedModelName || !stagedRiskPreset) {
      setError('Select a model and risk preset before starting the trader')
      return Promise.resolve()
    }
    return runControl(() =>
      startTrader({ modelName: stagedModelName, riskPresetId: stagedRiskPreset.id }),
    )
  }, [runControl, stagedModelName, stagedRiskPreset])

  const restart = useCallback(() => {
    if (!stagedModelName || !stagedRiskPreset) {
      setError('Select a model and risk preset before restarting the trader')
      return Promise.resolve()
    }
    return runControl(() =>
      restartTrader({ modelName: stagedModelName, riskPresetId: stagedRiskPreset.id }),
    )
  }, [runControl, stagedModelName, stagedRiskPreset])

  const stop = useCallback(() => runControl(() => stopTrader()), [runControl])
  const kill = useCallback(() => runControl(() => engageKillSwitch()), [runControl])
  const resume = useCallback(() => runControl(() => resumeKillSwitch()), [runControl])

  const runningSpec = status?.spec ?? null
  const stagedDiffersFromRunning =
    status?.running === true &&
    runningSpec !== null &&
    (runningSpec.model_name !== stagedModelName ||
      runningSpec.risk_preset_id !== (stagedRiskPreset?.id ?? null))

  return {
    status,
    activity,
    error,
    busy,
    stagedDiffersFromRunning,
    refresh,
    start,
    stop,
    restart,
    kill,
    resume,
  } as const
}
