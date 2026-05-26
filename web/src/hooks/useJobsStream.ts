import { useEffect, useState } from 'react'

import { listJobs, websocketUrl } from '@/lib/api'
import type { JobInfo } from '@/lib/types'

export function useJobsStream() {
  const [jobs, setJobs] = useState<JobInfo[]>([])

  useEffect(() => {
    let closed = false
    const socket = new WebSocket(websocketUrl('/api/jobs/stream'))

    socket.onmessage = (event) => {
      if (closed) return
      try {
        const payload = JSON.parse(event.data) as { jobs?: JobInfo[] }
        setJobs(payload.jobs ?? [])
      } catch {
        // Ignore malformed payloads and keep stream alive.
      }
    }

    socket.onerror = async () => {
      const fallbackJobs = await listJobs().catch(() => [])
      if (!closed) {
        setJobs(fallbackJobs)
      }
    }

    return () => {
      closed = true
      socket.close()
    }
  }, [])

  return jobs
}

