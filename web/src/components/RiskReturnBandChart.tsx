import { formatSigned } from '@/lib/format'
import type { RiskPresetTrial } from '@/lib/types'

export interface RiskReturnBandChartProps {
  readonly trials: readonly RiskPresetTrial[]
  readonly selectedRiskPresetId: string | null
}

const WIDTH = 760
const HEIGHT = 260
const PAD_X = 44
const PAD_Y = 30

export function RiskReturnBandChart({
  trials,
  selectedRiskPresetId,
}: RiskReturnBandChartProps) {
  if (trials.length === 0) {
    return null
  }
  const values = trials.flatMap((trial) => [
    trial.expected_return_per_market.p10,
    trial.expected_return_per_market.p25,
    trial.expected_return_per_market.expected,
    trial.expected_return_per_market.p75,
    trial.expected_return_per_market.p90,
  ])
  const minValue = Math.min(...values, 0)
  const maxValue = Math.max(...values, 0)
  const span = maxValue - minValue || 1
  const x = (index: number) =>
    trials.length === 1
      ? WIDTH / 2
      : PAD_X + (index * (WIDTH - PAD_X * 2)) / (trials.length - 1)
  const y = (value: number) => HEIGHT - PAD_Y - ((value - minValue) / span) * (HEIGHT - PAD_Y * 2)

  const line = (selector: (trial: RiskPresetTrial) => number) =>
    trials.map((trial, index) => `${x(index)},${y(selector(trial))}`).join(' ')
  const zeroY = y(0)

  return (
    <section className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Risk preset distribution
          </p>
          <h2 className="font-display text-lg font-semibold">Expected return per market</h2>
        </div>
        <div className="flex flex-wrap gap-3 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
          <LegendSwatch className="bg-foreground" label="Expected" />
          <LegendSwatch className="bg-red" label="P25/P10" />
          <LegendSwatch className="bg-cyan" label="P75/P90" />
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Expected return per market with percentile bands by risk preset"
          className="min-w-[680px]"
        >
          <line x1={PAD_X} x2={WIDTH - PAD_X} y1={zeroY} y2={zeroY} className="stroke-line" />
          <polyline
            points={line((trial) => trial.expected_return_per_market.p10)}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-red opacity-50"
          />
          <polyline
            points={line((trial) => trial.expected_return_per_market.p25)}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-red"
          />
          <polyline
            points={line((trial) => trial.expected_return_per_market.expected)}
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            className="text-foreground"
          />
          <polyline
            points={line((trial) => trial.expected_return_per_market.p75)}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-cyan"
          />
          <polyline
            points={line((trial) => trial.expected_return_per_market.p90)}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-cyan opacity-50"
          />
          {trials.map((trial, index) => {
            const pointX = x(index)
            const expectedY = y(trial.expected_return_per_market.expected)
            const selected = trial.risk_preset_id === selectedRiskPresetId
            return (
              <g key={trial.risk_preset_id}>
                <circle
                  cx={pointX}
                  cy={expectedY}
                  r={selected ? 5 : 3.5}
                  className={selected ? 'fill-foreground' : 'fill-muted'}
                />
                <text
                  x={pointX}
                  y={HEIGHT - 8}
                  textAnchor="middle"
                  className={selected ? 'fill-foreground font-mono text-[10px]' : 'fill-muted font-mono text-[10px]'}
                >
                  {trial.label}
                </text>
              </g>
            )
          })}
          <text x={PAD_X} y={16} className="fill-muted font-mono text-[10px]">
            {formatSigned(maxValue)}
          </text>
          <text x={PAD_X} y={HEIGHT - 6} className="fill-muted font-mono text-[10px]">
            {formatSigned(minValue)}
          </text>
        </svg>
      </div>
    </section>
  )
}

function LegendSwatch({ className, label }: { readonly className: string; readonly label: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`h-2 w-5 rounded-sm ${className}`} />
      {label}
    </span>
  )
}
