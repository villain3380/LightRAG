import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  listTraces,
  getTrace,
  deleteTrace,
  clearTraces,
  type TraceListItem as TraceItem,
  type TraceDetail,
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Loader2Icon, Trash2Icon, XIcon, FileTextIcon } from 'lucide-react'

/** Compact trace list entry in the sidebar. */
function TraceListEntry({
  item,
  selected,
  onSelect,
}: {
  item: TraceItem
  selected: boolean
  onSelect: (id: string) => void
}) {
  const ts = item.timestamp ? new Date(item.timestamp).toLocaleString() : ''
  return (
    <button
      onClick={() => onSelect(item.trace_id)}
      className={`w-full text-left px-3 py-2 text-xs border-b border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors ${
        selected ? 'bg-emerald-50 dark:bg-emerald-900/30' : ''
      }`}
    >
      <div className="font-mono truncate text-gray-500 dark:text-gray-400">{item.trace_id}</div>
      <div className="truncate text-gray-700 dark:text-gray-200">{item.query_preview}</div>
      <div className="text-gray-400 dark:text-gray-500">
        {item.mode}
        {item.has_sparse ? ' · sparse' : ''}
        {ts ? ` · ${ts}` : ''}
      </div>
      <div className="text-gray-300 dark:text-gray-600 text-[10px]">{item.file_size} B</div>
    </button>
  )
}

/** A/B ranking columns displayed side-by-side for comparison. */
function DualRanking({
  dense,
  sparse,
  fused,
}: {
  dense?: TraceDetail['path_a_dense_ranking']
  sparse?: TraceDetail['path_b_sparse_ranking']
  fused?: TraceDetail['path_ab_fused_ranking']
}) {
  const { t } = useTranslation()

  const renderTable = (
    data: { rank: number; chunk_id: string; file_path: string; distance: number; content_preview?: string }[] | undefined,
    label: string,
    extraCol?: (item: any) => string,
  ) => {
    if (!data || data.length === 0) return <p className="text-xs text-gray-400">{t('traces.noData')}</p>
    return (
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-300 dark:border-gray-600 text-left">
            <th className="px-1">#</th>
            <th className="px-1">Chunk</th>
            {extraCol && <th className="px-1 text-center">{label} #</th>}
            <th className="px-1 text-right">Dist</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.chunk_id + r.rank} className="border-b border-gray-100 dark:border-gray-700">
              <td className="px-1 text-gray-400">{r.rank}</td>
              <td className="px-1 truncate max-w-[120px]" title={`${r.file_path}  ${r.chunk_id}`}>{r.file_path?.split('/').pop() || r.chunk_id}</td>
              {extraCol && <td className="px-1 text-center text-gray-500">{extraCol(r)}</td>}
              <td className="px-1 text-right font-mono text-gray-500">{r.distance?.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <div className="grid grid-cols-3 gap-2">
      <div>
        <h4 className="text-sm font-semibold mb-1">{t('traces.dense')}</h4>
        {renderTable(dense, 'dense')}
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-1">{t('traces.sparse')}</h4>
        {renderTable(sparse, 'sparse')}
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-1">{t('traces.fused')}</h4>
        {fused && fused.length > 0
          ? renderTable(
            fused.map((r) => ({
              rank: r.rank,
              chunk_id: r.chunk_id,
              file_path: r.file_path,
              distance: r.distance,
            })),
            'fused',
            (r) => {
              const f = fused.find((x) => x.chunk_id === r.chunk_id)
              return f ? `D${f.dense_rank ?? '-'} S${f.sparse_rank ?? '-'}` : ''
            },
          )
          : <p className="text-xs text-gray-400">{t('traces.noData')}</p>}
      </div>
    </div>
  )
}

export default function TracesView() {
  const { t } = useTranslation()
  const [traces, setTraces] = useState<TraceItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<TraceDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const fetchList = useCallback(async () => {
    setLoading(true)
    try {
      const res = await listTraces()
      console.log('[TracesView] listTraces response:', res)
      console.log('[TracesView] traces array:', res?.traces)
      if (res && res.traces) {
        setTraces(res.traces)
      } else {
        console.warn('[TracesView] no traces field in response, setting empty')
        setTraces([])
      }
    } catch (err) {
      console.error('[TracesView] fetch failed:', err)
      toast.error(t('traces.errors.fetchListFailed', { error: errorMessage(err) }))
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    fetchList()
  }, [fetchList])

  const selectTrace = useCallback(
    async (id: string) => {
      setSelectedId(id)
      setDetailLoading(true)
      setDetail(null)
      try {
        const data = await getTrace(id)
        setDetail(data)
      } catch (err) {
        toast.error(t('traces.errors.fetchDetailFailed', { error: errorMessage(err) }))
        setSelectedId(null)
      } finally {
        setDetailLoading(false)
      }
    },
    [t],
  )

  const closeDetail = useCallback(() => {
    setSelectedId(null)
    setDetail(null)
  }, [])

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteTrace(id)
        if (selectedId === id) closeDetail()
        fetchList()
      } catch (err) {
        toast.error(t('traces.errors.deleteFailed', { error: errorMessage(err) }))
      }
    },
    [selectedId, closeDetail, fetchList, t],
  )

  const handleClearAll = useCallback(async () => {
    if (!window.confirm(t('traces.clearAllConfirm'))) return
    try {
      await clearTraces()
      closeDetail()
      fetchList()
      toast.success(t('traces.clearAllDone'))
    } catch (err) {
      toast.error(t('traces.errors.clearAllFailed', { error: errorMessage(err) }))
    }
  }, [closeDetail, fetchList, t])

  return (
    <div className="flex h-full">
      {/* Left sidebar: trace list */}
      <div className="w-[280px] shrink-0 border-r border-gray-200 dark:border-gray-700 flex flex-col">
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-700">
          <span className="text-sm font-semibold">{t('traces.title')}</span>
          <div className="flex gap-1">
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={fetchList} title={t('traces.refresh')} disabled={loading}>
              <Loader2Icon className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleClearAll} title={t('traces.clearAll')}>
              <Trash2Icon className="h-3 w-3" />
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-auto">
          {(traces || []).length === 0 ? (
            <p className="p-4 text-xs text-gray-400">{t('traces.empty')}</p>
          ) : (
            traces.map((item) => (
              <TraceListEntry
                key={item.trace_id}
                item={item}
                selected={item.trace_id === selectedId}
                onSelect={selectTrace}
              />
            ))
          )}
        </div>
      </div>

      {/* Main detail area */}
      <div className="flex-1 overflow-auto p-4">
        {!selectedId && <p className="text-gray-400 text-sm mt-8 text-center">{t('traces.selectHint')}</p>}

        {detailLoading && (
          <div className="flex items-center justify-center mt-20">
            <Loader2Icon className="h-6 w-6 animate-spin text-gray-400" />
          </div>
        )}

        {detail && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold">{detail.query}</h3>
                <p className="text-xs text-gray-400">
                  {detail.trace_id} · {detail.mode}
                  {detail.weights ? ` · D${detail.weights.dense} S${detail.weights.sparse}` : ''}
                  {' · '}
                  {detail.timestamp ? new Date(detail.timestamp).toLocaleString() : ''}
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleDelete(detail.trace_id)} title={t('traces.delete')}>
                  <Trash2Icon className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={closeDetail} title={t('traces.close')}>
                  <XIcon className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Keywords */}
            {detail.keywords && (
              <Card className="mb-4">
                <CardHeader className="py-2 px-3"><CardTitle className="text-sm">{t('traces.keywords')}</CardTitle></CardHeader>
                <CardContent className="py-1 px-3 text-xs">
                  <p>{t('traces.highLevel')}: {detail.keywords.high_level?.join(', ') || '-'}</p>
                  <p>{t('traces.lowLevel')}: {detail.keywords.low_level?.join(', ') || '-'}</p>
                </CardContent>
              </Card>
            )}

            {/* A/B/Fused rankings */}
            {(detail.path_a_dense_ranking || detail.path_b_sparse_ranking || detail.path_ab_fused_ranking) && (
              <Card className="mb-4">
                <CardHeader className="py-2 px-3"><CardTitle className="text-sm">{t('traces.vectorRankings')}</CardTitle></CardHeader>
                <CardContent className="py-1 px-3">
                  <DualRanking
                    dense={detail.path_a_dense_ranking}
                    sparse={detail.path_b_sparse_ranking}
                    fused={detail.path_ab_fused_ranking}
                  />
                </CardContent>
              </Card>
            )}

            {/* Final context */}
            {detail.final_context && (
              <Card className="mb-4">
                <CardHeader className="py-2 px-3">
                  <CardTitle className="text-sm">
                    {t('traces.finalContext')}
                    {' · '}
                    {detail.final_context.total_chunks_kept ?? detail.final_context.vector_chunks?.length}{' '}
                    {t('traces.chunksSent')}
                  </CardTitle>
                </CardHeader>
                <CardContent className="py-1 px-3 text-xs">
                  {detail.final_context.vector_chunks?.map((c: any, i: number) => (
                    <div key={c.chunk_id || i} className="flex items-center gap-2 py-0.5 border-b border-gray-100 dark:border-gray-700 last:border-0">
                      <FileTextIcon className="h-3 w-3 text-gray-400 shrink-0" />
                      <span className="truncate" title={`${c.file_path}  ${c.chunk_id}`}>
                        {c.file_path?.split('/').pop() || c.chunk_id}
                      </span>
                      <span className="text-gray-400 shrink-0 ml-auto">{c.content_preview?.substring(0, 60)}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
