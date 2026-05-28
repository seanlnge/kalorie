import { useEffect, useMemo, useRef, useState } from 'react'

import { computeRiskTrial } from '@/lib/api'
import type { RiskPreset, RiskPresetTrial, SavedModelMetadata } from '@/lib/types'

export function useRiskPresetTrial(
  model: SavedModelMetadata | null,
  riskPreset: RiskPreset | null,
) {
  const [computedTrial, setComputedTrial] = useState<RiskPresetTrial | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestId = useRef(0)

  const bundledTrial = useMemo(() => {
    if (!model?.risk_preset_trials.length) return null
    if (!riskPreset) return model.risk_preset_trials[0] ?? null
    return model.risk_preset_trials.find((trial) => trial.risk_preset_id === riskPreset.id) ?? null
  }, [model, riskPreset])

  useEffect(() => {
    requestId.current += 1
    const currentRequest = requestId.current
    setError(null)
    setComputedTrial(null)
    if (!model || !riskPreset || (bundledTrial && isCompleteTrial(bundledTrial))) {
      setLoading(false)
      return
    }
    setLoading(true)
    computeRiskTrial(model.name, riskPreset)
      .then((trial) => {
        if (requestId.current === currentRequest) {
          setComputedTrial(trial)
        }
      })
      .catch((err: unknown) => {
        if (requestId.current === currentRequest) {
          setError(err instanceof Error ? err.message : 'Failed to compute risk preset trial')
        }
      })
      .finally(() => {
        if (requestId.current === currentRequest) {
          setLoading(false)
        }
      })
  }, [bundledTrial, model, riskPreset])

  return {
    trial: bundledTrial && isCompleteTrial(bundledTrial) ? bundledTrial : computedTrial ?? bundledTrial,
    bundled: Boolean(bundledTrial && isCompleteTrial(bundledTrial)),
    loading,
    error,
  } as const
}

function isCompleteTrial(trial: RiskPresetTrial): boolean {
  return (
    trial.return_variance_per_market !== undefined &&
    Boolean(trial.roi_projection?.length) &&
    Boolean(trial.roi_paths?.length)
  )
}
