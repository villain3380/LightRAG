import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { XIcon, FileTextIcon, CopyIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ReferenceItem } from '@/api/lightrag'
import Button from '@/components/ui/Button'

interface ChunkSourceViewProps {
  citation: ReferenceItem
  onClose: () => void
}

/** Right-pane chunk viewer shown when an inline `[^n]` citation is clicked. */
export default function ChunkSourceView({ citation, onClose }: ChunkSourceViewProps) {
  const { t } = useTranslation()
  const content = citation.content?.[0] ?? ''
  const fileName = citation.file_path?.split('/').pop() || citation.file_path || ''
  // order_index is 0-based; display as 1-based "第 N 个"
  const chunkNumber =
    typeof citation.order_index === 'number' ? citation.order_index + 1 : null

  const copyChunkId = () => {
    if (!citation.chunk_id) return
    navigator.clipboard
      .writeText(citation.chunk_id)
      .then(() => toast.success(t('retrievePanel.citation.copied')))
      .catch(() => toast.error(t('retrievePanel.citation.copyFailed')))
  }

  return (
    <div className="flex h-full flex-col rounded-lg border bg-primary-foreground/60">
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <FileTextIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold">
            {t('retrievePanel.citation.source')} [^{citation.reference_id}]
          </div>
          <div className="truncate text-xs text-muted-foreground" title={citation.file_path}>
            {fileName}
          </div>
          <div className="mt-0.5 flex items-center gap-1 text-[10px] text-muted-foreground/80">
            {chunkNumber !== null && (
              <span className="shrink-0">
                {t('retrievePanel.citation.chunkNumber', { n: chunkNumber })}
              </span>
            )}
            {citation.chunk_id && (
              <>
                <span className="truncate font-mono" title={citation.chunk_id}>
                  {citation.chunk_id}
                </span>
                <button
                  type="button"
                  onClick={copyChunkId}
                  className="shrink-0 cursor-pointer hover:text-foreground"
                  title={t('retrievePanel.citation.copyChunkId')}
                >
                  <CopyIcon className="h-3 w-3" />
                </button>
              </>
            )}
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={onClose}
          title={t('retrievePanel.citation.backToSettings')}
        >
          <XIcon className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 overflow-auto p-3 text-sm">
        {content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        ) : (
          <p className="text-xs text-muted-foreground">
            {t('retrievePanel.citation.noContent')}
          </p>
        )}
      </div>
    </div>
  )
}
