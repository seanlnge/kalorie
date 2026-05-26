import { Route, Routes } from 'react-router-dom'

import { RightRailJobs } from '@/components/jobs/RightRailJobs'
import { Toaster } from '@/components/ui/sonner'
import { useJobsStream } from '@/hooks/useJobsStream'
import { HomePage } from '@/pages/HomePage'
import { MarketPage } from '@/pages/MarketPage'

function App() {
  const jobs = useJobsStream()

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex max-w-[1600px] flex-col md:flex-row">
        <main className="flex-1 p-6">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/markets/:ticker" element={<MarketPage />} />
          </Routes>
        </main>
        <RightRailJobs jobs={jobs} />
      </div>
      <Toaster richColors />
    </div>
  )
}

export default App
