import type { JobInfo } from '@/lib/types'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface RightRailJobsProps {
  jobs: JobInfo[]
}

export function RightRailJobs({ jobs }: RightRailJobsProps) {
  return (
    <aside className="w-full border-l bg-muted/20 p-4 md:w-80">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
        Background Jobs
      </h2>
      <div className="space-y-3">
        {jobs.length === 0 ? (
          <Card>
            <CardContent className="pt-4 text-sm text-muted-foreground">
              No jobs running. Submit a market run to start.
            </CardContent>
          </Card>
        ) : (
          jobs.map((job) => (
            <Card key={job.job_id} size="sm">
              <CardHeader>
                <CardTitle className="text-sm">{job.market_ticker}</CardTitle>
                <CardDescription className="font-mono text-xs">
                  {job.status}
                  {job.wait_reason ? ` · ${job.wait_reason}` : ''}
                </CardDescription>
              </CardHeader>
            </Card>
          ))
        )}
      </div>
    </aside>
  )
}

