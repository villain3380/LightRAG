import { useState, useEffect, useCallback } from 'react'
import { toast } from 'sonner'
import {
  listTraceTurns,
  getTurnTraces,
  getTraceChunks,
  getChunkContent,
  type TraceTurn,
  type QueryTraceRow,
  type QueryTraceChunk,
  type ChunkContent,
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import Button from '@/components/ui/Button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import { Loader2Icon, RefreshCwIcon } from 'lucide-react'

/** 检索路径追踪页面：turn -> trace -> chunk 三级下钻，点 chunk 按需取正文。
 *  数据来自 lightrag-server 的 /query_trace/*（public.query_trace* 表）。*/
const SOURCE_COLORS: Record<string, string> = {
  dense: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  sparse: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300',
  entity: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  relation: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
}

function SourceBadges({ sources }: { sources: string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {(sources || []).map((s) => (
        <span key={s} className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${SOURCE_COLORS[s] || 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'}`}>
          {s}
        </span>
      ))}
    </div>
  )
}

function ms(v: number | null | undefined): string {
  if (v == null) return '-'
  return `${Math.round(v)}ms`
}

function Loader() {
  return (
    <div className="flex items-center justify-center mt-10">
      <Loader2Icon className="h-5 w-5 animate-spin text-gray-400" />
    </div>
  )
}

const BORDER = 'border-gray-200 dark:border-gray-700'
const HOVER = 'hover:bg-gray-100 dark:hover:bg-gray-800'
const SEL = 'bg-emerald-50 dark:bg-emerald-900/30'

export default function TracesView() {
  const [turns, setTurns] = useState<TraceTurn[]>([])
  const [turnsLoading, setTurnsLoading] = useState(true)
  const [selTurn, setSelTurn] = useState<string | null>(null)
  const [traces, setTraces] = useState<QueryTraceRow[]>([])
  const [tracesLoading, setTracesLoading] = useState(false)
  const [selTrace, setSelTrace] = useState<string | null>(null)
  const [selTraceRow, setSelTraceRow] = useState<QueryTraceRow | null>(null)
  const [chunks, setChunks] = useState<QueryTraceChunk[]>([])
  const [chunksLoading, setChunksLoading] = useState(false)
  const [chunkDialog, setChunkDialog] = useState<{ chunkId: string; loading: boolean; data: ChunkContent | null } | null>(null)

  const fetchTurns = useCallback(async () => {
    setTurnsLoading(true)
    try {
      setTurns(await listTraceTurns(50))
    } catch (e) {
      toast.error(`加载 turn 失败: ${errorMessage(e)}`)
    } finally {
      setTurnsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchTurns()
  }, [fetchTurns])

  const selectTurn = useCallback(async (turnId: string) => {
    setSelTurn(turnId)
    setSelTrace(null)
    setSelTraceRow(null)
    setChunks([])
    setTracesLoading(true)
    try {
      setTraces(await getTurnTraces(turnId))
    } catch (e) {
      toast.error(`加载 trace 失败: ${errorMessage(e)}`)
    } finally {
      setTracesLoading(false)
    }
  }, [])

  const selectTrace = useCallback(async (row: QueryTraceRow) => {
    setSelTrace(row.trace_id)
    setSelTraceRow(row)
    setChunksLoading(true)
    try {
      setChunks(await getTraceChunks(row.trace_id))
    } catch (e) {
      toast.error(`加载 chunk 失败: ${errorMessage(e)}`)
    } finally {
      setChunksLoading(false)
    }
  }, [])

  const openChunk = useCallback(async (chunkId: string) => {
    setChunkDialog({ chunkId, loading: true, data: null })
    try {
      const data = await getChunkContent(chunkId)
      setChunkDialog({ chunkId, loading: false, data })
    } catch (e) {
      toast.error(`取 chunk 正文失败: ${errorMessage(e)}`)
      setChunkDialog(null)
    }
  }, [])

  return (
    <div className="flex h-full text-sm">
      {/* col1: turns */}
      <div className={`w-[260px] shrink-0 border-r ${BORDER} flex flex-col`}>
        <div className={`flex items-center justify-between px-3 py-2 border-b ${BORDER}`}>
          <span className="font-semibold">Turn 列表</span>
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={fetchTurns} disabled={turnsLoading} title="刷新">
            <RefreshCwIcon className={`h-3 w-3 ${turnsLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        <div className="flex-1 overflow-auto">
          {turns.length === 0 ? (
            <p className="p-4 text-xs text-gray-400">无数据（走一次 /query 后刷新）</p>
          ) : (
            turns.map((t) => (
              <button
                key={t.turn_id}
                onClick={() => selectTurn(t.turn_id)}
                className={`w-full text-left px-3 py-2 text-xs border-b ${BORDER} ${HOVER} transition-colors ${selTurn === t.turn_id ? SEL : ''}`}
              >
                <div className="truncate text-gray-700 dark:text-gray-200">{t.original_query}</div>
                <div className="text-gray-400 text-[10px]">
                  {t.source || '-'} · {t.trace_count} trace · {t.chunk_count} chunk
                </div>
                <div className="text-gray-400 text-[10px]">
                  {t.latest_created_at ? new Date(t.latest_created_at).toLocaleString() : ''}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* col2: traces */}
      <div className={`w-[300px] shrink-0 border-r ${BORDER} flex flex-col`}>
        <div className={`px-3 py-2 border-b ${BORDER} font-semibold`}>
          Trace{selTurn ? ` · ${selTurn.slice(0, 14)}` : ''}
        </div>
        <div className="flex-1 overflow-auto">
          {!selTurn ? (
            <p className="p-4 text-xs text-gray-400">点击左侧 turn</p>
          ) : tracesLoading ? (
            <Loader />
          ) : traces.length === 0 ? (
            <p className="p-4 text-xs text-gray-400">无 trace</p>
          ) : (
            traces.map((r) => (
              <button
                key={r.trace_id}
                onClick={() => selectTrace(r)}
                className={`w-full text-left px-3 py-2 text-xs border-b ${BORDER} ${HOVER} transition-colors ${selTrace === r.trace_id ? SEL : ''}`}
              >
                <div className="truncate text-gray-700 dark:text-gray-200">
                  #{r.call_index ?? '-'} {r.search_query}
                </div>
                <div className="text-gray-400 text-[10px]">
                  {r.mode} · {r.source}
                  {r.cache_hit ? ' · cache' : ''}
                  {r.status !== 'success' ? ` · ${r.status}` : ''}
                </div>
                <div className="text-gray-500 text-[10px] font-mono">
                  rw {ms(r.rewrite_ms)} · ret {ms(r.retrieval_ms)} · rr {ms(r.rerank_ms)} · gen {ms(r.generation_ms)}
                </div>
                {r.mode !== 'naive' && r.kg_entity_ms != null && (
                  <div className="text-gray-400 text-[10px] font-mono">
                    kg: emb {ms(r.kg_embed_ms)} ent {ms(r.kg_entity_ms)} rel {ms(r.kg_relation_ms)} chunk {ms(r.kg_chunk_ms)}
                  </div>
                )}
              </button>
            ))
          )}
        </div>
      </div>

      {/* col3: chunks */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className={`px-3 py-2 border-b ${BORDER}`}>
          <span className="font-semibold">Chunk 明细</span>
          {selTraceRow && (
            <span className="text-xs text-gray-400 ml-2">
              final {selTraceRow.chunks_final ?? '-'}/{selTraceRow.chunks_after_rerank ?? '-'} after rerank · {selTraceRow.chunks_retrieved ?? '-'} retrieved
            </span>
          )}
        </div>
        <div className="flex-1 overflow-auto p-2">
          {!selTrace ? (
            <p className="text-xs text-gray-400 mt-8 text-center">点击中间 trace 查看 chunk 溯源</p>
          ) : chunksLoading ? (
            <Loader />
          ) : chunks.length === 0 ? (
            <p className="text-xs text-gray-400">无 chunk</p>
          ) : (
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className={`border-b ${BORDER} text-left text-gray-500`}>
                  <th className="px-1 py-1">重排后#</th>
                  <th className="px-1">重排前#</th>
                  <th className="px-1">来源</th>
                  <th className="px-1">分数</th>
                  <th className="px-1 text-center">最终</th>
                  <th className="px-1">chunk</th>
                </tr>
              </thead>
              <tbody>
                {chunks.map((c) => (
                  <tr
                    key={c.id}
                    onClick={() => openChunk(c.chunk_id)}
                    className={`border-b ${BORDER} cursor-pointer ${HOVER}`}
                  >
                    <td className="px-1 py-1 font-mono">{c.post_rerank_rank ?? '-'}</td>
                    <td className="px-1 font-mono text-gray-400">{c.pre_rerank_position ?? '-'}</td>
                    <td className="px-1"><SourceBadges sources={c.sources} /></td>
                    <td className="px-1 font-mono text-gray-500">{c.rerank_score?.toFixed(3) ?? '-'}</td>
                    <td className="px-1 text-center">
                      {c.in_final_context ? <span className="text-emerald-600">✓</span> : <span className="text-gray-300">·</span>}
                    </td>
                    <td className="px-1 truncate max-w-[160px] text-gray-500" title={c.chunk_id}>
                      {c.file_path?.split('/').pop() || c.chunk_id.slice(0, 12)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* chunk content dialog (按需取正文) */}
      <Dialog open={!!chunkDialog} onOpenChange={(o) => !o && setChunkDialog(null)}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="text-sm">Chunk 正文 · {chunkDialog?.chunkId.slice(0, 16)}</DialogTitle>
          </DialogHeader>
          {chunkDialog?.loading ? (
            <Loader />
          ) : (
            <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-gray-50 dark:bg-gray-900 p-3 rounded">
              {chunkDialog?.data?.content || '(空)'}
            </pre>
          )}
          {chunkDialog?.data?.file_path && (
            <div className="text-[10px] text-gray-400 mt-2 break-all">{chunkDialog.data.file_path}</div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
