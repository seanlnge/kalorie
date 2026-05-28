import { formatProbability, formatSigned } from '@/lib/format'
import type { RiskPresetTrial, RiskReturnPathPoint, RiskReturnProjectionPoint } from '@/lib/types'

export interface RiskReturnBandChartProps {
  readonly trial: RiskPresetTrial | null
  readonly loading?: boolean
}

const WIDTH = 760
const HEIGHT = 300
const PAD_LEFT = 74
const PAD_RIGHT = 28
const PAD_TOP = 34
const PAD_BOTTOM = 48

export function RiskReturnBandChart({ trial, loading = false }: RiskReturnBandChartProps) {
  if (!trial) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-panel/60 p-5 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          Selected model + risk preset
        </p>
        <h2 className="mt-2 font-display text-lg font-semibold">
          {loading ? 'Computing trial distribution...' : 'No trial distribution yet'}
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          {loading
            ? 'The app is running this risk preset across saved evaluation rows and building a bootstrap risk-return distribution.'
            : 'This preset can score live markets, but the model card does not include a bootstrap risk-return distribution for it yet.'}
        </p>
      </section>
    )
  }

  const projection = normalizeProjection(trial)
  const roiPaths = normalizePaths(trial)
  const maxMarkets = Math.max(...projection.map((point) => point.market_count), 1)
  const yValues = projection.flatMap((point) => [
    point.roi.p10,
    point.roi.p25,
    point.roi.expected,
    point.roi.p75,
    point.roi.p90,
    0,
  ])
  yValues.push(...roiPaths.flatMap((path) => path.map((point) => point.roi)))
  const minValue = Math.min(...yValues)
  const maxValue = Math.max(...yValues)
  const ySpan = maxValue - minValue || 1
  const x = (markets: number) =>
    PAD_LEFT + (markets / maxMarkets) * (WIDTH - PAD_LEFT - PAD_RIGHT)
  const y = (value: number) => PAD_TOP + ((maxValue - value) / ySpan) * (HEIGHT - PAD_TOP - PAD_BOTTOM)
  const zeroY = y(0)
  const finalExpected = projection[projection.length - 1]?.roi.expected

  return (
    <section className="rounded-lg border border-line bg-panel/82 p-4 shadow-terminal">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
            Selected model + risk preset
          </p>
          <h2 className="font-display text-lg font-semibold">
            {trial.label} Monte Carlo ROI projection
          </h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">
            Simulated future markets start at 0% bankroll ROI, then sample historical model+preset
            trade outcomes with replacement. X is market count; Y is cumulative ROI / bankroll.
          </p>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
          <Summary label={`EV/${maxMarkets} mkts`} value={formatSigned(finalExpected)} />
          <Summary label="EV/10 mkts" value={formatSigned(trial.ev_per_10_markets)} />
          <Summary label="Trade %" value={formatProbability(trial.trade_percent)} />
          <Summary label="Risk of ruin" value={formatProbability(trial.risk_of_ruin_estimate)} />
          <Summary label="Variance" value={formatVariance(trial.return_variance_per_market)} />
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="ROI by number of markets with percentile bands for the selected risk preset"
          className="min-w-[680px]"
        >
          <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={zeroY} y2={zeroY} className="stroke-line" />
          <line x1={PAD_LEFT} x2={PAD_LEFT} y1={PAD_TOP} y2={HEIGHT - PAD_BOTTOM} className="stroke-line" />
          <line
            x1={PAD_LEFT}
            x2={WIDTH - PAD_RIGHT}
            y1={HEIGHT - PAD_BOTTOM}
            y2={HEIGHT - PAD_BOTTOM}
            className="stroke-line"
          />
          <text
            x={PAD_LEFT - 44}
            y={PAD_TOP + 8}
            textAnchor="middle"
            className="fill-muted font-mono text-[9px] uppercase tracking-[0.12em]"
            transform={`rotate(-90 ${PAD_LEFT - 44} ${PAD_TOP + 8})`}
          >
            ROI
          </text>
          <text
            x={(PAD_LEFT + WIDTH - PAD_RIGHT) / 2}
            y={HEIGHT - 10}
            textAnchor="middle"
            className="fill-muted font-mono text-[9px] uppercase tracking-[0.12em]"
          >
            Number of markets
          </text>
          {axisTicks(minValue, maxValue).map((tick) => (
            <g key={tick}>
              <line x1={PAD_LEFT - 4} x2={PAD_LEFT} y1={y(tick)} y2={y(tick)} className="stroke-line" />
              <text x={PAD_LEFT - 8} y={y(tick) + 3} textAnchor="end" className="fill-muted font-mono text-[9px]">
                {formatProbability(tick)}
              </text>
            </g>
          ))}
          {projection.map((point) => (
            <g key={point.market_count}>
              <line
                x1={x(point.market_count)}
                x2={x(point.market_count)}
                y1={HEIGHT - PAD_BOTTOM}
                y2={HEIGHT - PAD_BOTTOM + 4}
                className="stroke-line"
              />
              <text
                x={x(point.market_count)}
                y={HEIGHT - PAD_BOTTOM + 18}
                textAnchor="middle"
                className="fill-muted font-mono text-[9px]"
              >
                {point.market_count}
              </text>
            </g>
          ))}
          {roiPaths.map((path, index) => (
            <polyline
              key={index}
              points={pathPolyline(path, x, y)}
              fill="none"
              strokeWidth="0.75"
              className="stroke-foreground opacity-[0.12]"
            />
          ))}
          <polyline points={polyline(projection, x, y, 'p10')} fill="none" strokeWidth="1" className="stroke-red opacity-50" />
          <polyline points={polyline(projection, x, y, 'p25')} fill="none" strokeWidth="1" className="stroke-red" />
          <polyline points={polyline(projection, x, y, 'expected')} fill="none" strokeWidth="1.5" className="stroke-foreground" />
          <polyline points={polyline(projection, x, y, 'p75')} fill="none" strokeWidth="1" className="stroke-cyan" />
          <polyline points={polyline(projection, x, y, 'p90')} fill="none" strokeWidth="1" className="stroke-cyan opacity-50" />
          <ProjectionLabel projection={projection} x={x} y={y} label="Expected" bandKey="expected" offsetY={-10} />
          <ProjectionLabel projection={projection} x={x} y={y} label="P10" bandKey="p10" offsetY={14} />
          <ProjectionLabel projection={projection} x={x} y={y} label="P90" bandKey="p90" offsetY={-8} />
        </svg>
      </div>
    </section>
  )
}

function Summary({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <p>{label}</p>
      <p className="text-foreground">{value}</p>
    </div>
  )
}

function ProjectionLabel({
  projection,
  x,
  y,
  label,
  bandKey,
  offsetY,
}: {
  readonly projection: readonly RiskReturnProjectionPoint[]
  readonly x: (marketCount: number) => number
  readonly y: (value: number) => number
  readonly label: string
  readonly bandKey: keyof RiskReturnProjectionPoint['roi']
  readonly offsetY: number
}) {
  const last = projection[projection.length - 1]
  const value = last.roi[bandKey]
  return (
    <text x={x(last.market_count)} y={y(value) + offsetY} textAnchor="end" className="fill-muted font-mono text-[9px]">
      {label} {formatProbability(value)}
    </text>
  )
}

function polyline(
  points: readonly RiskReturnProjectionPoint[],
  x: (markets: number) => number,
  y: (value: number) => number,
  key: keyof RiskReturnProjectionPoint['roi'],
): string {
  return points.map((point) => `${x(point.market_count)},${y(point.roi[key])}`).join(' ')
}

function pathPolyline(
  points: readonly RiskReturnPathPoint[],
  x: (markets: number) => number,
  y: (value: number) => number,
): string {
  return points.map((point) => `${x(point.market_count)},${y(point.roi)}`).join(' ')
}

function axisTicks(minValue: number, maxValue: number): number[] {
  return [...new Set([minValue, 0, maxValue].map((value) => Number(value.toFixed(4))))]
}

function formatVariance(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(6) : '--'
}

function normalizePaths(trial: RiskPresetTrial): RiskReturnPathPoint[][] {
  return trial.roi_paths?.length ? trial.roi_paths : []
}

function normalizeProjection(trial: RiskPresetTrial): RiskReturnProjectionPoint[] {
  if (trial.roi_projection?.length) {
    return trial.roi_projection
  }
  return [
    {
      market_count: 0,
      roi: { p10: 0, p25: 0, expected: 0, p75: 0, p90: 0 },
    },
    {
      market_count: Math.max(1, trial.market_count),
      roi: trial.expected_return_per_market,
    },
  ]
}
