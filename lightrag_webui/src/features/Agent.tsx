import { useState } from 'react'

const AGENT_API = 'http://localhost:9956'

interface Message {
  role: 'user' | 'agent'
  text: string
}

/**
 * Agent 页面：content 临时框（左侧）+ agent 聊天框（右侧）。
 *
 * 流程：
 * 1. 左侧粘贴 content -> [暂存] -> POST /api/content -> 得到 #编号
 * 2. 右侧输入"存一下 #1" -> POST /api/chat -> agent read_content + 生成字段 + ingest_insight
 *
 * content 不经 agent output（ingest_insight 传 content_id，agent 不复述 content）。
 */
export default function Agent() {
  const [content, setContent] = useState('')
  const [contentId, setContentId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const storeContent = async () => {
    if (!content) return
    try {
      const r = await fetch(`${AGENT_API}/api/content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      })
      const data = await r.json()
      if (data.content_id) {
        setContentId(data.content_id)
      } else {
        alert(`暂存失败: ${data.error || '未知错误'}`)
      }
    } catch (e: any) {
      alert(`暂存失败（agent server 没起?）: ${e.message}`)
    }
  }

  const chat = async () => {
    if (!input.trim() || loading) return
    const userMsg = input
    const newMessages = [...messages, { role: 'user' as const, text: userMsg }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)
    try {
      const r = await fetch(`${AGENT_API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
      })
      const data = await r.json()
      setMessages([...newMessages, { role: 'agent' as const, text: data.response }])
    } catch (e: any) {
      setMessages([
        ...newMessages,
        { role: 'agent' as const, text: `错误: ${e.message}` },
      ])
    }
    setLoading(false)
  }

  return (
    <div className="flex h-full gap-4 p-4">
      {/* 左：content 临时框 */}
      <div className="flex w-1/2 flex-col gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold">Content 临时框</h3>
          {contentId !== null && (
            <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800">
              #{contentId}
            </span>
          )}
        </div>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="粘贴 content 全文（agent 不会复述这段，只读取生成 summary）..."
          className="flex-1 resize-none rounded border p-2 text-sm"
        />
        <button
          onClick={storeContent}
          disabled={!content}
          className="rounded bg-emerald-500 px-4 py-1.5 text-sm text-white disabled:opacity-50"
        >
          暂存 {contentId !== null && `→ #${contentId}`}
        </button>
      </div>

      {/* 右：agent 聊天 */}
      <div className="flex w-1/2 flex-col gap-2">
        <h3 className="text-sm font-bold">Agent 对话</h3>
        <div className="flex-1 overflow-auto rounded border p-2 text-sm">
          {messages.length === 0 && (
            <div className="text-gray-400">
              先在左侧暂存 content，然后输入"存一下 #1"让 agent 入库。
              <br />
              也可以直接问问题，如"查一下长鑫相关的"。
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`mb-2 ${m.role === 'user' ? 'text-right' : ''}`}>
              <span
                className={`inline-block max-w-[80%] whitespace-pre-wrap rounded px-2 py-1 text-left ${
                  m.role === 'user'
                    ? 'bg-blue-100 text-blue-800'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {m.text}
              </span>
            </div>
          ))}
          {loading && <div className="text-gray-400">agent 思考中...</div>}
        </div>
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                chat()
              }
            }}
            placeholder={
              contentId !== null ? `如：存一下 #${contentId}` : '先暂存 content...'
            }
            className="flex-1 rounded border px-2 py-1 text-sm"
            disabled={loading}
          />
          <button
            onClick={chat}
            disabled={loading || !input.trim()}
            className="rounded bg-emerald-500 px-4 py-1 text-sm text-white disabled:opacity-50"
          >
            发送
          </button>
        </div>
      </div>
    </div>
  )
}
