import { FileText } from 'lucide-react'

import type { SavedModelMetadata } from '@/lib/types'

export interface ModelDetailsDrawerProps {
  readonly model: SavedModelMetadata | null
}

export function ModelDetailsDrawer({ model }: ModelDetailsDrawerProps) {
  if (!model) {
    return null
  }

  return (
    <details className="rounded-3xl border border-line/70 bg-panel/70 p-5">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-2xl border border-cyan/30 bg-cyan/10 text-cyan">
            <FileText size={18} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-muted">Model README / Details</p>
            <h2 className="font-display text-lg font-semibold">{model.name}</h2>
          </div>
        </div>
        <span className="font-mono text-xs text-muted">expand</span>
      </summary>
      <div className="mt-5 grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-2xl border border-line/70 bg-background/70 p-4 font-mono text-xs leading-6 text-muted">
          {model.readme}
        </pre>
        <div className="rounded-2xl border border-line/70 bg-background/50 p-4">
          <p className="mb-3 text-xs uppercase tracking-[0.22em] text-muted">Artifact paths</p>
          <div className="space-y-2">
            {Object.entries(model.artifact_paths).map(([name, path]) => (
              <div key={name} className="rounded-xl border border-line/50 bg-panel/50 p-3">
                <p className="text-[10px] uppercase tracking-wide text-muted">{name}</p>
                <p className="mt-1 break-all font-mono text-xs text-foreground">{path}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </details>
  )
}
