import { useEffect, useMemo, useRef, useState } from 'react'

import { computeRiskTrial } from '@/lib/api'
import type { RiskPreset, RiskPresetTrial, SavedModelMetadata } from '@/lib/types'

export function useRiskPresetTrials(
  model: SavedModelMetadata | null,
  presets: readonly RiskPreset[],
) {
  const [computedTrials, setComputedTrials] = useState<Map<string, RiskPresetTrial>>(new Map())
  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set())
  const [error, setError] = useState<string | null>(null)
  const requestId = useRef(0)

  const bundledTrials = useMemo(() => {
    return new Map((model?.risk_preset_trials ?? []).map((trial) => [trial.risk_preset_id, trial]))
  }, [model])

  useEffect(() => {
    requestId.current += 1
    const currentRequest = requestId.current
    setComputedTrials(new Map())
    setError(null)

    if (!model || presets.length === 0) {
      setLoadingIds(new Set())
      return
    }

    const missingPresets = presets.filter((preset) => {
      const bundled = bundledTrials.get(preset.id)
      return !bundled || !isCompleteTrial(bundled)
    })
    setLoadingIds(new Set(missingPresets.map((preset) => preset.id)))

    if (missingPresets.length === 0) {
      return
    }

    Promise.allSettled(missingPresets.map((preset) => computeRiskTrial(model.name, preset)))
      .then((results) => {
        if (requestId.current !== currentRequest) return
        const nextTrials = new Map<string, RiskPresetTrial>()
        const rejected = results.find((result) => result.status === 'rejected')
        for (const result of results) {
          if (result.status === 'fulfilled') {
            nextTrials.set(result.value.risk_preset_id, result.value)
          }
        }
        setComputedTrials(nextTrials)
        if (rejected?.status === 'rejected') {
          setError(
            rejected.reason instanceof Error
              ? rejected.reason.message
              : 'Failed to compute one or more risk preset trials',
          )
        }
      })
      .finally(() => {
        if (requestId.current === currentRequest) {
          setLoadingIds(new Set())
        }
      })
  }, [bundledTrials, model, presets])

  const trialsByPresetId = useMemo(() => {
    return new Map([...bundledTrials.entries(), ...computedTrials.entries()])
  }, [bundledTrials, computedTrials])

  return { trialsByPresetId, loadingIds, error } as const
}

function isCompleteTrial(trial: RiskPresetTrial): boolean {
  return (
    trial.return_variance_per_market !== undefined &&
    Boolean(trial.roi_projection?.length) &&
    Boolean(trial.roi_paths?.length)
  )
}
