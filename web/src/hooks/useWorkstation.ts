import { useEffect, useMemo, useRef, useState } from 'react'

import { getModel, getSampleRows, listModels, scoreModel } from '@/lib/api'
import type { ExecutionMode, SampleRow, SavedModelMetadata, ScoreRow } from '@/lib/types'

interface WorkstationState {
  models: SavedModelMetadata[]
  selectedModel: SavedModelMetadata | null
  selectedModelName: string | null
  sampleRows: SampleRow[]
  selectedRowIndex: number
  executionMode: ExecutionMode
  predictions: ScoreRow[]
  loading: boolean
  scoring: boolean
  error: string | null
}

interface WorkstationActions {
  selectModel: (modelName: string) => void
  setExecutionMode: (mode: ExecutionMode) => void
  setSelectedRowIndex: (rowIndex: number) => void
  scoreSelectedRow: () => Promise<void>
  scoreUploadedCsv: (file: File) => Promise<void>
}

export type UseWorkstationResult = Readonly<WorkstationState & WorkstationActions>

export function useWorkstation(): UseWorkstationResult {
  const [models, setModels] = useState<SavedModelMetadata[]>([])
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<SavedModelMetadata | null>(null)
  const [sampleRows, setSampleRows] = useState<SampleRow[]>([])
  const [selectedRowIndex, setSelectedRowIndex] = useState(0)
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('all')
  const [predictions, setPredictions] = useState<ScoreRow[]>([])
  const [loading, setLoading] = useState(true)
  const [scoring, setScoring] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scoreRequestId = useRef(0)
  const detailRequestId = useRef(0)

  useEffect(() => {
    let cancelled = false
    listModels()
      .then((nextModels) => {
        if (cancelled) return
        setModels(nextModels)
        setSelectedModelName((current) => current ?? nextModels[0]?.name ?? null)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load saved models')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedModelName) {
      setSelectedModel(null)
      return
    }
    let cancelled = false
    const requestId = detailRequestId.current + 1
    detailRequestId.current = requestId
    setError(null)
    scoreRequestId.current += 1
    setScoring(false)
    setSelectedModel(null)
    setSampleRows([])
    setSelectedRowIndex(0)
    setPredictions([])
    Promise.all([getModel(selectedModelName), getSampleRows(selectedModelName)])
      .then(([model, rows]) => {
        if (cancelled || detailRequestId.current !== requestId) return
        scoreRequestId.current += 1
        setScoring(false)
        setSelectedModel(model)
        setSampleRows(rows)
        setSelectedRowIndex(Number(rows[0]?.row_index ?? 0))
        setPredictions([])
      })
      .catch((err: unknown) => {
        if (!cancelled && detailRequestId.current === requestId) {
          setError(err instanceof Error ? err.message : 'Failed to load model detail')
        }
      })
    return () => {
      cancelled = true
    }
  }, [selectedModelName])

  const actions = useMemo<WorkstationActions>(
    () => ({
      selectModel: (modelName) => {
        if (modelName === selectedModelName) return
        detailRequestId.current += 1
        scoreRequestId.current += 1
        setScoring(false)
        setSelectedModel(null)
        setSampleRows([])
        setSelectedRowIndex(0)
        setPredictions([])
        setSelectedModelName(modelName)
      },
      setExecutionMode: (mode) => {
        scoreRequestId.current += 1
        setExecutionMode(mode)
        setScoring(false)
        setPredictions([])
      },
      setSelectedRowIndex: (rowIndex) => {
        scoreRequestId.current += 1
        setSelectedRowIndex(rowIndex)
        setScoring(false)
        setPredictions([])
      },
      scoreSelectedRow: async () => {
        if (!selectedModelName || !selectedModel) return
        const requestId = scoreRequestId.current + 1
        scoreRequestId.current = requestId
        setScoring(true)
        setError(null)
        try {
          const response = await scoreModel({
            modelName: selectedModelName,
            executionMode,
            rowIndex: selectedRowIndex,
          })
          if (scoreRequestId.current === requestId) {
            setPredictions(response.rows)
          }
        } catch (err) {
          if (scoreRequestId.current === requestId) {
            setError(err instanceof Error ? err.message : 'Failed to score sample row')
          }
        } finally {
          if (scoreRequestId.current === requestId) {
            setScoring(false)
          }
        }
      },
      scoreUploadedCsv: async (file: File) => {
        if (!selectedModelName || !selectedModel) return
        const requestId = scoreRequestId.current + 1
        scoreRequestId.current = requestId
        setScoring(true)
        setError(null)
        try {
          const response = await scoreModel({
            modelName: selectedModelName,
            executionMode,
            rowIndex: selectedRowIndex,
            csvFile: file,
          })
          if (scoreRequestId.current === requestId) {
            setPredictions(response.rows)
          }
        } catch (err) {
          if (scoreRequestId.current === requestId) {
            setError(err instanceof Error ? err.message : 'Failed to score uploaded CSV')
          }
        } finally {
          if (scoreRequestId.current === requestId) {
            setScoring(false)
          }
        }
      },
    }),
    [executionMode, selectedModel, selectedModelName, selectedRowIndex],
  )

  return {
    models,
    selectedModel,
    selectedModelName,
    sampleRows,
    selectedRowIndex,
    executionMode,
    predictions,
    loading,
    scoring,
    error,
    ...actions,
  }
}
