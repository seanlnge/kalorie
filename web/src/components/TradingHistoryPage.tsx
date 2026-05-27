export function TradingHistoryPage() {
  return (
    <section className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <HistoryMetric label="Execution" value="Not live" tone="text-amber" />
        <HistoryMetric label="Executed trades" value="--" />
        <HistoryMetric label="Open positions" value="--" />
        <HistoryMetric label="Realized PnL" value="--" />
      </div>

      <section className="rounded-lg border border-line bg-panel/82 p-5 shadow-terminal">
        <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">
          Executed trade ledger
        </p>
        <h2 className="mt-2 font-display text-lg font-semibold">No trades have been placed yet</h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Poll snapshots and risk overlays are model tests, not executed trades. This ledger will stay
          empty until trading functionality records real fills, positions, and realized PnL.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <RoadmapStep label="1" title="Market scan" description="Current Markets ranks live opportunities." />
          <RoadmapStep label="2" title="Human review" description="Use model + risk preset output as research." />
          <RoadmapStep label="3" title="Execution later" description="Real fills will populate this ledger." />
        </div>
      </section>
    </section>
  )
}

function HistoryMetric({
  label,
  value,
  tone = 'text-foreground',
}: {
  readonly label: string
  readonly value: string
  readonly tone?: string
}) {
  return (
    <div className="rounded-lg border border-line bg-panel/75 p-4 shadow-terminal">
      <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className={`mt-2 break-all font-mono text-lg font-semibold leading-6 ${tone}`}>{value}</p>
    </div>
  )
}

function RoadmapStep({
  label,
  title,
  description,
}: {
  readonly label: string
  readonly title: string
  readonly description: string
}) {
  return (
    <div className="rounded-md border border-line bg-background/55 p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-cyan">Step {label}</p>
      <p className="mt-2 font-display text-sm font-semibold text-foreground">{title}</p>
      <p className="mt-1 text-xs leading-5 text-muted">{description}</p>
    </div>
  )
}
