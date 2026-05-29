import { Activity, OctagonX, Pause, Play, RotateCcw } from 'lucide-react'

import { formatDollars } from '@/lib/format'
import type {
  OpenPositionsSummary,
  RiskPreset,
  TraderActivityItem,
  TraderStatus,
} from '@/lib/types'

interface AutoTraderPageProps {
  readonly status: TraderStatus | null
  readonly activity: TraderActivityItem[]
  readonly error: string | null
  readonly busy: boolean
  readonly stagedModelName: string | null
  readonly stagedRiskPreset: RiskPreset | null
  readonly stagedDiffersFromRunning: boolean
  readonly positions: OpenPositionsSummary
  readonly onStart: () => void
  readonly onStop: () => void
  readonly onRestart: () => void
  readonly onKill: () => void
  readonly onResume: () => void
}

const ACTION_TONE: Record<string, string> = {
  submitted: 'text-green',
  dry_run_approved: 'text-cyan',
  rejected: 'text-muted',
  halted: 'text-amber',
  submit_failed: 'text-red',
  duplicate_skipped: 'text-muted',
  kill_switch: 'text-red',
  portfolio_refresh: 'text-cyan',
  rescore_requested: 'text-amber',
  rescore_failed: 'text-red',
}

export function AutoTraderPage({
  status,
  activity,
  error,
  busy,
  stagedModelName,
  stagedRiskPreset,
  stagedDiffersFromRunning,
  positions,
  onStart,
  onStop,
  onRestart,
  onKill,
  onResume,
}: AutoTraderPageProps) {
  const running = status?.running ?? false
  const killActive = status?.kill_switch_active ?? false
  const mode = status?.mode ?? 'off'

  return (
    <div className="space-y-3">
      <StatusBar
        status={status}
        running={running}
        mode={mode}
        stagedModelName={stagedModelName}
        stagedRiskPresetId={stagedRiskPreset?.id ?? null}
        stagedDiffersFromRunning={stagedDiffersFromRunning}
      />

      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-panel/60 px-3 py-2">
        <ControlButton
          label="Start"
          icon={<Play size={13} />}
          tone="green"
          disabled={busy || running}
          onClick={onStart}
        />
        <ControlButton
          label="Stop"
          icon={<Pause size={13} />}
          tone="muted"
          disabled={busy || !running}
          onClick={onStop}
        />
        <ControlButton
          label="Restart with changes"
          icon={<RotateCcw size={13} />}
          tone={stagedDiffersFromRunning ? 'cyan' : 'muted'}
          disabled={busy || !running || !stagedDiffersFromRunning}
          onClick={onRestart}
        />
        <div className="ml-auto">
          {killActive ? (
            <ControlButton
              label="Clear kill switch"
              icon={<Play size={13} />}
              tone="amber"
              disabled={busy}
              onClick={onResume}
            />
          ) : (
            <ControlButton
              label="Emergency stop"
              icon={<OctagonX size={13} />}
              tone="red"
              disabled={busy}
              onClick={onKill}
            />
          )}
        </div>
      </div>

      {stagedDiffersFromRunning ? (
        <div className="rounded-lg border border-cyan/30 bg-cyan/10 px-3 py-2 font-mono text-[11px] text-cyan">
          Staged config ({stagedModelName} / {stagedRiskPreset?.id}) differs from the running bot (
          {status?.spec?.model_name} / {status?.spec?.risk_preset_id}). Use "Restart with changes" to
          apply.
        </div>
      ) : null}

      {status?.startup_error ? (
        <div className="rounded-lg border border-amber/35 bg-amber/10 px-3 py-2 font-mono text-[11px] text-amber">
          Start-up re-score did not complete: {status.startup_error}. The bot skips any signal not
          scored by the running model until a fresh score lands.
        </div>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-red/35 bg-red/10 px-3 py-2 font-mono text-[11px] text-red">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1.6fr_1fr]">
        <ActivityFeed activity={activity} />
        <PositionsPanel positions={positions} />
      </div>
    </div>
  )
}

interface StatusBarProps {
  readonly status: TraderStatus | null
  readonly running: boolean
  readonly mode: string
  readonly stagedModelName: string | null
  readonly stagedRiskPresetId: string | null
  readonly stagedDiffersFromRunning: boolean
}

function StatusBar({ status, running, mode, stagedModelName, stagedRiskPresetId }: StatusBarProps) {
  const summary = status?.last_summary ?? null
  return (
    <div className="rounded-xl border border-line bg-panel/60 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={[
            'inline-flex h-7 items-center gap-2 rounded-lg border px-3 font-mono text-[10px] font-semibold uppercase tracking-[0.16em]',
            running
              ? 'border-green/40 bg-green/10 text-green'
              : 'border-line bg-panelStrong text-muted',
          ].join(' ')}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${running ? 'bg-green animate-pulse' : 'bg-muted'}`}
          />
          {running ? 'Running' : 'Stopped'}
        </span>
        <ModePill mode={mode} />
        {status?.kill_switch_active ? (
          <span className="inline-flex h-7 items-center gap-2 rounded-lg border border-red/40 bg-red/10 px-3 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-red">
            Kill switch engaged
          </span>
        ) : null}
        <div className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[11px]">
          <Metric label="Running model" value={status?.spec?.model_name ?? '--'} />
          <Metric label="Running preset" value={status?.spec?.risk_preset_id ?? '--'} />
          <Metric label="Passes" value={String(status?.pass_count ?? 0)} />
          <Metric label="Orders today" value={String(status?.daily_orders ?? 0)} />
          <Metric label="P&L today" value={formatDollars(status?.daily_loss ?? 0)} />
        </div>
      </div>
      {summary ? (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line/60 pt-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
          <span>last pass</span>
          <span className="text-foreground">evaluated {summary.evaluated}</span>
          <span className="text-green">submitted {summary.submitted}</span>
          <span className="text-cyan">dry-run {summary.dry_run_approved}</span>
          <span>rejected {summary.rejected}</span>
          <span className="text-amber">halted {summary.halted}</span>
          <span className="text-red">failed {summary.failed}</span>
        </div>
      ) : (
        <div className="mt-2 border-t border-line/60 pt-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
          Staged: {stagedModelName ?? '--'} / {stagedRiskPresetId ?? '--'}
        </div>
      )}
    </div>
  )
}

function ModePill({ mode }: { readonly mode: string }) {
  const tone =
    mode === 'live'
      ? 'border-red/40 bg-red/10 text-red'
      : mode === 'dry_run'
        ? 'border-cyan/30 bg-cyan/10 text-cyan'
        : 'border-line bg-panelStrong text-muted'
  return (
    <span
      className={`inline-flex h-7 items-center rounded-lg border px-3 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] ${tone}`}
    >
      {mode === 'live' ? 'Live orders' : mode === 'dry_run' ? 'Dry run' : 'Mode off'}
    </span>
  )
}

function Metric({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-[9px] uppercase tracking-[0.2em] text-muted">{label}</span>
      <span className="text-foreground">{value}</span>
    </span>
  )
}

interface ControlButtonProps {
  readonly label: string
  readonly icon: React.ReactNode
  readonly tone: 'green' | 'cyan' | 'amber' | 'red' | 'muted'
  readonly disabled?: boolean
  readonly onClick: () => void
}

function ControlButton({ label, icon, tone, disabled, onClick }: ControlButtonProps) {
  const toneClass: Record<ControlButtonProps['tone'], string> = {
    green: 'border-green/50 text-green hover:bg-green/10',
    cyan: 'border-cyan/50 text-cyan hover:bg-cyan/10',
    amber: 'border-amber/50 text-amber hover:bg-amber/10',
    red: 'border-red/50 text-red hover:bg-red/10',
    muted: 'border-line text-muted hover:bg-panel hover:text-foreground',
  }
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        'inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] transition',
        toneClass[tone],
        disabled ? 'cursor-not-allowed opacity-40' : '',
      ].join(' ')}
    >
      {icon}
      {label}
    </button>
  )
}

function ActivityFeed({ activity }: { readonly activity: TraderActivityItem[] }) {
  const ordered = [...activity].reverse()
  return (
    <section className="rounded-xl border border-line bg-panel/60">
      <header className="flex items-center gap-2 border-b border-line px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan">
        <Activity size={13} />
        Activity feed
      </header>
      <div className="max-h-[28rem] overflow-auto">
        {ordered.length === 0 ? (
          <p className="px-3 py-6 text-center font-mono text-[11px] text-muted">
            No trader activity yet.
          </p>
        ) : (
          <table className="w-full border-collapse font-mono text-[11px]">
            <thead className="sticky top-0 bg-panelStrong text-[9px] uppercase tracking-[0.16em] text-muted">
              <tr>
                <th className="px-2 py-1.5 text-left">Time</th>
                <th className="px-2 py-1.5 text-left">Event</th>
                <th className="px-2 py-1.5 text-left">Market</th>
                <th className="px-2 py-1.5 text-right">Side</th>
                <th className="px-2 py-1.5 text-right">Qty</th>
                <th className="px-2 py-1.5 text-right">Limit</th>
                <th className="px-2 py-1.5 text-left">Reason</th>
              </tr>
            </thead>
            <tbody>
              {ordered.map((item, index) => (
                <tr key={`${item.ts ?? index}-${index}`} className="border-t border-line/50">
                  <td className="px-2 py-1.5 text-muted">{formatClock(item.ts)}</td>
                  <td className={`px-2 py-1.5 ${ACTION_TONE[item.event] ?? 'text-foreground'}`}>
                    {item.event}
                  </td>
                  <td className="px-2 py-1.5 text-foreground">{item.market_ticker ?? '--'}</td>
                  <td className="px-2 py-1.5 text-right">{item.side ?? '--'}</td>
                  <td className="px-2 py-1.5 text-right">{item.order_contracts ?? '--'}</td>
                  <td className="px-2 py-1.5 text-right">
                    {item.limit_price ? item.limit_price.toFixed(2) : '--'}
                  </td>
                  <td className="px-2 py-1.5 text-muted">{item.reason ?? item.detail ?? '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}

function PositionsPanel({ positions }: { readonly positions: OpenPositionsSummary }) {
  return (
    <section className="rounded-xl border border-line bg-panel/60">
      <header className="border-b border-line px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan">
        Open positions
      </header>
      <div className="max-h-[28rem] overflow-auto">
        {positions.positions.length === 0 ? (
          <p className="px-3 py-6 text-center font-mono text-[11px] text-muted">
            {positions.available ? 'No open positions.' : 'Kalshi account not connected.'}
          </p>
        ) : (
          <table className="w-full border-collapse font-mono text-[11px]">
            <thead className="bg-panelStrong text-[9px] uppercase tracking-[0.16em] text-muted">
              <tr>
                <th className="px-2 py-1.5 text-left">Market</th>
                <th className="px-2 py-1.5 text-right">Side</th>
                <th className="px-2 py-1.5 text-right">Qty</th>
                <th className="px-2 py-1.5 text-right">Avg</th>
                <th className="px-2 py-1.5 text-right">Exposure</th>
              </tr>
            </thead>
            <tbody>
              {positions.positions.map((position) => (
                <tr key={position.market_ticker} className="border-t border-line/50">
                  <td className="px-2 py-1.5 text-foreground">{position.market_ticker}</td>
                  <td className="px-2 py-1.5 text-right">{position.side}</td>
                  <td className="px-2 py-1.5 text-right">{position.contracts}</td>
                  <td className="px-2 py-1.5 text-right">
                    {position.average_price ? position.average_price.toFixed(2) : '--'}
                  </td>
                  <td className="px-2 py-1.5 text-right">{formatDollars(position.exposure ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}

function formatClock(ts: string | undefined): string {
  if (!ts) return '--'
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) return ts
  return date.toLocaleTimeString('en-US', { hour12: false })
}
