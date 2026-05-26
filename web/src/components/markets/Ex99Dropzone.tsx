import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface Ex99DropzoneProps {
  onSubmit: (files: File[]) => Promise<void>
}

export function Ex99Dropzone({ onSubmit }: Ex99DropzoneProps) {
  const [files, setFiles] = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)

  const canSubmit = useMemo(() => files.length > 0 && !submitting, [files.length, submitting])

  function addFiles(fileList: FileList | null) {
    if (!fileList) return
    const next = Array.from(fileList).filter((file) => /ex-99/i.test(file.name))
    if (next.length === 0) return
    setFiles((current) => {
      const deduped = new Map<string, File>()
      for (const file of [...current, ...next]) deduped.set(file.name, file)
      return Array.from(deduped.values())
    })
  }

  async function handleSubmit() {
    setSubmitting(true)
    try {
      await onSubmit(files)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Upload EX-99.* files</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <label
          className="flex min-h-28 cursor-pointer items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 px-4 text-center text-sm text-muted-foreground"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            addFiles(event.dataTransfer.files)
          }}
        >
          <input
            type="file"
            className="hidden"
            multiple
            onChange={(event) => addFiles(event.target.files)}
          />
          Drag and drop EX-99 files here, or click to choose files
        </label>
        <div className="space-y-1 text-sm">
          {files.length === 0 ? (
            <p className="text-muted-foreground">No files selected.</p>
          ) : (
            files.map((file) => (
              <div key={file.name} className="flex items-center justify-between rounded border px-2 py-1">
                <span>{file.name}</span>
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => setFiles((current) => current.filter((item) => item.name !== file.name))}
                >
                  Remove
                </Button>
              </div>
            ))
          )}
        </div>
        <Button disabled={!canSubmit} onClick={handleSubmit}>
          {submitting ? 'Submitting...' : 'Train for this market'}
        </Button>
      </CardContent>
    </Card>
  )
}

