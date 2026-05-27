import { formatProbability, formatSigned } from '@/lib/format'
import type { RiskPresetTrial } from '@/lib/types'

export interface RiskReturnBandChartProps {
  readonly trial: RiskPresetTrial | null
}

const WIDTH = 760
const HEIGHT = 260
const PAD_X = 64
const PAD_Y = 30

export function RiskReturnBandChart({ trial }: RiskReturnBandChartProps) {
  if (!trial) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-panel/60 p-5 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          Selected model + risk preset
        </p>
        <h2 className="mt-2 font-display text-lg font-semibold">No trial distribution yet</h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          This preset can score live markets, but the model card does not include a bootstrap
          risk-return distribution for it yet. Built-in presets show percentile bands here; custom
          session presets need a generated trial artifact before this chart can compare downside and
          upside.
        </p>
      </section>
    )
  }

  const band = trial.expected_return_per_market
  const values = [band.p10, band.p25, band.expected, band.p75, band.p90]
  const minValue = Math.min(...values, 0)
  const maxValue = Math.max(...values, 0)
  const span = maxValue - minValue || 1
  const x = (value: number) => PAD_X + ((value - minValue) / span) * (WIDTH - PAD_X * 2)
  const axisY = HEIGHT / 2
  const zeroX = x(0)

  return (
    <section className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Selected model + risk preset
          </p>
          <h2 className="font-display text-lg font-semibold">
            {trial.label} risk-return distribution
          </h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">
            White marks expected return. Red is downside percentile range, blue is upside
            percentile range, and the vertical rail marks zero return.
          </p>
        </div>
        <div className="grid grid-cols-3 gap-2 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
          <Summary label="EV/10 mkts" value={formatSigned(trial.ev_per_10_markets)} />
          <Summary label="Trade %" value={formatProbability(trial.trade_percent)} />
          <Summary label="Risk of ruin" value={formatProbability(trial.risk_of_ruin_estimate)} />
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Expected return per market with percentile bands for the selected risk preset"
          className="min-w-[680px]"
        >
          <line x1={PAD_X} x2={WIDTH - PAD_X} y1={axisY} y2={axisY} className="stroke-line" />
          <line x1={zeroX} x2={zeroX} y1={PAD_Y} y2={HEIGHT - PAD_Y} className="stroke-line" />
          <text
            x={zeroX}
            y={PAD_Y - 8}
            textAnchor="middle"
            className="fill-muted font-mono text-[9px] uppercase tracking-[0.12em]"
          >
            zero
          </text>
          <line
            x1={x(band.p10)}
            x2={x(band.p25)}
            y1={axisY}
            y2={axisY}
            strokeWidth="8"
            strokeLinecap="round"
            className="stroke-red opacity-50"
          />
          <line
            x1={x(band.p25)}
            x2={x(band.expected)}
            y1={axisY}
            y2={axisY}
            strokeWidth="8"
            strokeLinecap="round"
            className="stroke-red"
          />
          <line
            x1={x(band.expected)}
            x2={x(band.p75)}
            y1={axisY}
            y2={axisY}
            strokeWidth="8"
            strokeLinecap="round"
            className="stroke-cyan"
          />
          <line
            x1={x(band.p75)}
            x2={x(band.p90)}
            y1={axisY}
            y2={axisY}
            strokeWidth="8"
            strokeLinecap="round"
            className="stroke-cyan opacity-50"
          />
          <circle cx={x(band.expected)} cy={axisY} r="7" className="fill-foreground" />
          <BandLabel x={x(band.p10)} y={axisY + 32} label="P10 downside" value={band.p10} />
          <BandLabel x={x(band.p25)} y={axisY - 24} label="P25" value={band.p25} />
          <BandLabel x={x(band.expected)} y={axisY - 54} label="Expected" value={band.expected} />
          <BandLabel x={x(band.p75)} y={axisY - 24} label="P75" value={band.p75} />
          <BandLabel x={x(band.p90)} y={axisY + 32} label="P90 upside" value={band.p90} />
          <text x={PAD_X} y={HEIGHT - 8} className="fill-muted font-mono text-[10px]">
            {formatSigned(minValue)}
          </text>
          <text
            x={WIDTH - PAD_X}
            y={HEIGHT - 8}
            textAnchor="end"
            className="fill-muted font-mono text-[10px]"
          >
            {formatSigned(maxValue)}
          </text>
        </svg>
      </div>
    </section>
  )
}

function Summary({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div className="rounded border border-line bg-background/55 px-2 py-1">
      <p>{label}</p>
      <p className="mt-1 text-foreground">{value}</p>
    </div>
  )
}

function BandLabel({
  x,
  y,
  label,
  value,
}: {
  readonly x: number
  readonly y: number
  readonly label: string
  readonly value: number
}) {
  return (
    <text x={x} y={y} textAnchor="middle" className="fill-muted font-mono text-[10px]">
      {label} {formatSigned(value)}
    </text>
  )
}
