import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { XIcon, Loader2Icon, FileTextIcon, LayersIcon } from 'lucide-react'

import Button from '@/components/ui/Button'
import { getDocumentChunks, type DocStatusResponse, type DocumentChunksResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'

interface ChunkInspectorProps {
  doc: DocStatusResponse
  onClose: () => void
}

export default function ChunkInspector({ doc, onClose }: ChunkInspectorProps) {
  const { t } = useTranslation()
  const [data, setData] = useState<DocumentChunksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const fetchData = async () => {
      try {
        const result = await getDocumentChunks(doc.id)
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          const msg = errorMessage(err)
          setError(msg)
          toast.error(t('documentPanel.chunkInspector.errors.fetchFailed', { error: msg }))
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchData()
    return () => {
      cancelled = true
    }
  }, [doc.id, t])

  // Close on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    // Lock body scroll while the inspector is open
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', handleKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  const meta = data?.doc_metadata
  const chunks = data?.chunks ?? []
  const displayTitle = doc.file_path || doc.id

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background">
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
        <FileTextIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="truncate font-medium" title={displayTitle}>{displayTitle}</span>
        <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
          {loading ? '…' : `${chunks.length} ${t('documentPanel.chunkInspector.chunksUnit')}`}
        </span>
        <Button variant="ghost" size="icon" className="ml-auto h-8 w-8" onClick={onClose} title={t('documentPanel.chunkInspector.close')}>
          <XIcon className="h-4 w-4" />
        </Button>
      </header>

      {/* Body: left metadata / right chunk cards */}
      <div className="flex min-h-0 flex-1">
        {/* Left: document-level metadata */}
        <aside className="w-[28%] min-w-[240px] max-w-[360px] shrink-0 overflow-y-auto border-r p-3 text-sm">
          <div className="mb-2 flex items-center gap-1.5 font-semibold">
            <LayersIcon className="h-4 w-4" />
            {t('documentPanel.chunkInspector.docMetadata')}
          </div>
          {loading ? (
            <div className="text-muted-foreground">{t('documentPanel.chunkInspector.loading')}</div>
          ) : meta ? (
            <dl className="space-y-1.5">
              <MetaRow label={t('documentPanel.chunkInspector.fields.id')} value={meta.id} mono />
              <MetaRow label={t('documentPanel.chunkInspector.fields.filePath')} value={meta.file_path} />
              <MetaRow label={t('documentPanel.chunkInspector.fields.status')} value={meta.status} />
              <MetaRow label={t('documentPanel.chunkInspector.fields.contentLength')} value={meta.content_length?.toLocaleString()} />
              <MetaRow label={t('documentPanel.chunkInspector.fields.chunksCount')} value={(meta.chunks_count ?? 0).toString()} />
              <MetaRow label={t('documentPanel.chunkInspector.fields.trackId')} value={meta.track_id || '-'} mono />
              <MetaRow label={t('documentPanel.chunkInspector.fields.contentHash')} value={meta.content_hash || '-'} mono />
              <MetaRow label={t('documentPanel.chunkInspector.fields.createdAt')} value={meta.created_at ? new Date(meta.created_at).toLocaleString() : '-'} />
              <MetaRow label={t('documentPanel.chunkInspector.fields.updatedAt')} value={meta.updated_at ? new Date(meta.updated_at).toLocaleString() : '-'} />
              {meta.error_msg ? (
                <MetaRow label={t('documentPanel.chunkInspector.fields.errorMsg')} value={meta.error_msg} />
              ) : null}
              {meta.metadata && Object.keys(meta.metadata).length > 0 ? (
                <div className="pt-1">
                  <div className="text-xs text-muted-foreground">{t('documentPanel.chunkInspector.fields.metadata')}</div>
                  <pre className="mt-1 overflow-x-auto rounded bg-muted p-2 text-xs">{JSON.stringify(meta.metadata, null, 2)}</pre>
                </div>
              ) : null}
            </dl>
          ) : null}
        </aside>

        {/* Right: chunk cards */}
        <main className="min-w-0 flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
              {t('documentPanel.chunkInspector.loading')}
            </div>
          ) : error ? (
            <div className="flex h-full items-center justify-center text-sm text-red-500">{error}</div>
          ) : chunks.length === 0 ? (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              {t('documentPanel.chunkInspector.empty')}
            </div>
          ) : (
            <div className="mx-auto max-w-4xl space-y-3">
              {chunks.map((chunk) => (
                <div key={chunk.chunk_id} className="rounded-md border">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/40 px-3 py-1.5 text-xs text-muted-foreground">
                    <span className="font-mono">chunk id: {chunk.chunk_id.slice(0, 16)}</span>
                    <span>
                      {t('documentPanel.chunkInspector.order')} {chunk.order_index} · {t('documentPanel.chunkInspector.tokens')} {chunk.tokens}
                    </span>
                  </div>
                  {chunk.heading && typeof chunk.heading === 'object' && (
                    <div className="px-3 pt-2 text-xs text-muted-foreground">
                      {t('documentPanel.chunkInspector.heading')} {JSON.stringify(chunk.heading)}
                    </div>
                  )}
                  <div className="px-3 py-2">
                    <div className="prose dark:prose-invert max-w-none break-words text-sm prose-headings:mt-2 prose-headings:mb-1 prose-p:my-1 prose-table:my-2 prose-th:border prose-th:border-border prose-td:border prose-td:border-border prose-th:px-2 prose-th:py-1 prose-td:px-2 prose-td:py-1">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                        {chunk.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

function MetaRow({ label, value, mono }: { label: string; value: string | undefined; mono?: boolean }) {
  return (
    <div className="flex flex-col">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={`break-all ${mono ? 'font-mono text-xs' : 'text-sm'}`}>{value ?? '-'}</dd>
    </div>
  )
}
