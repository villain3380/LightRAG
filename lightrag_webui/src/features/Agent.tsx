import { useCallback, useEffect, useRef, useState } from 'react'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { ChatMessage, type MessageWithError } from '@/components/retrieval/ChatMessage'
import { useDebounce } from '@/hooks/useDebounce'
import { throttle } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { copyToClipboard } from '@/utils/clipboard'
import { CopyIcon, EraserIcon, PlusIcon, SendIcon, SquareIcon, TrashIcon } from 'lucide-react'
import { toast } from 'sonner'

const AGENT_API = 'http://localhost:9956'

// Helper function to generate unique IDs with browser compatibility (mirrors RetrievalView)
const generateUniqueId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

interface ContentBox {
  key: number
  text: string
  storedId: number | null
}

/**
 * Agent 页面（基于 RetrievalView 布局，风格/体验统一）：
 * - 左侧：agent 聊天（SSE 流式 + markdown，复用 ChatMessage）
 * - 右侧：content 临时框列表（可增删，编号 + 复制 + 暂存）
 * - 拖拽分栏
 *
 * 后端契约（agent/dp_server.ts）：POST /api/content -> {content_id}；
 * POST /api/chat -> SSE data:<delta> ... data:[DONE]
 */
export default function Agent() {
  // 当前 tab 是否激活（控制 ChatMessage 的 loading 指示与 thinking 透明度）
  const currentTab = useSettingsStore.use.currentTab()
  const isAgentTabActive = currentTab === 'agent'

  // ── content 临时框 ──
  const [contents, setContents] = useState<ContentBox[]>([
    { key: 1, text: '', storedId: null },
  ])
  const [nextKey, setNextKey] = useState(2)

  // ── 聊天 ──
  const [messages, setMessages] = useState<MessageWithError[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  // 发送后短暂禁用 Stop 按钮，防止双击误 abort 刚启动的查询（参考 RetrievalView）
  const [stopDisabled, setStopDisabled] = useState(false)
  const stopCooldownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // 智能输入：单行用 Input，多行用 Textarea（参考 RetrievalView）
  const hasMultipleLines = input.includes('\n')
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null)

  // 滚动跟随（参考 RetrievalView）
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const shouldFollowScrollRef = useRef(true)
  const programmaticScrollRef = useRef(false)
  const isFormInteractionRef = useRef(false)
  const isReceivingResponseRef = useRef(false)

  // 在飞请求的 abort controller + 当前 assistant 消息 id
  const abortRef = useRef<AbortController | null>(null)
  const activeAssistantIdRef = useRef<string | null>(null)

  // 响应计时（client-side stopwatch -> assistantMessage.responseTime，ChatMessage 自动显示）
  const responseTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const responseStartRef = useRef<number | null>(null)

  // ── 拖拽分栏（参考 RetrievalView） ──
  const [rightPct, setRightPct] = useState(40)
  const startDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const onMove = (ev: MouseEvent) => {
      const pct = ((window.innerWidth - ev.clientX) / window.innerWidth) * 100
      setRightPct(Math.min(85, Math.max(15, pct)))
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  // 滚动到底（参考 RetrievalView：programmatic 标记 + auto 行为）
  const scrollToBottom = useCallback(() => {
    programmaticScrollRef.current = true
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
    })
  }, [])

  // 统一 textarea 高度调整
  const adjustTextareaHeight = useCallback((element: HTMLTextAreaElement) => {
    requestAnimationFrame(() => {
      element.style.height = 'auto'
      element.style.height = Math.min(element.scrollHeight, 120) + 'px'
    })
  }, [])

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInput(e.target.value)
    },
    []
  )

  // agent 聊天（SSE 流式接收 + markdown，消息结构对齐 RetrievalView 的 MessageWithError）
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!input.trim() || loading) return

      const userMsg = input

      const userMessage: MessageWithError = {
        id: generateUniqueId(),
        content: userMsg,
        role: 'user',
      }
      const assistantMessage: MessageWithError = {
        id: generateUniqueId(),
        content: '',
        role: 'assistant', // ChatMessage 只认 user/assistant/system，agent 回复映射成 assistant
        responseTime: 0,
      }

      setMessages((prev) => [...prev, userMessage, assistantMessage])
      setInput('')
      setLoading(true)
      shouldFollowScrollRef.current = true
      isReceivingResponseRef.current = true
      activeAssistantIdRef.current = assistantMessage.id

      // Stop 按钮冷却（防双击）
      setStopDisabled(true)
      if (stopCooldownTimerRef.current) clearTimeout(stopCooldownTimerRef.current)
      stopCooldownTimerRef.current = setTimeout(() => setStopDisabled(false), 500)

      // 重置输入框高度
      if (inputRef.current && 'style' in inputRef.current) {
        inputRef.current.style.height = '40px'
      }

      // 启动响应计时（每 100ms 更新 assistantMessage.responseTime）
      responseStartRef.current = Date.now()
      assistantMessage.responseTime = 0
      if (responseTimerRef.current) clearInterval(responseTimerRef.current)
      responseTimerRef.current = setInterval(() => {
        const elapsed = (Date.now() - (responseStartRef.current ?? Date.now())) / 1000
        const rounded = parseFloat(elapsed.toFixed(1))
        assistantMessage.responseTime = rounded
        setMessages((prev) => {
          const newMsgs = [...prev]
          const last = newMsgs[newMsgs.length - 1]
          if (last && last.id === assistantMessage.id) {
            last.responseTime = rounded
          }
          return newMsgs
        })
      }, 100)

      setTimeout(() => scrollToBottom(), 0)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const r = await fetch(`${AGENT_API}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg }),
          signal: controller.signal,
        })
        if (!r.body) throw new Error('无响应体')
        const reader = r.body.getReader()
        const decoder = new TextDecoder()
        let agentText = ''
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (line.startsWith('data:')) {
              const data = line.slice(5).trim()
              if (data === '[DONE]') break
              // 后端 delta 是纯文本；JSON.parse 仅在恰好是合法 JSON 字符串时生效，否则原样使用
              let delta: string = data
              try {
                const parsed = JSON.parse(data)
                delta = typeof parsed === 'string' ? parsed : data
              } catch {
                // 非 JSON，直接用
              }
              agentText += delta
              setMessages((prev) => {
                const newMsgs = [...prev]
                const last = newMsgs[newMsgs.length - 1]
                if (last && last.id === assistantMessage.id) {
                  last.content = agentText
                }
                return newMsgs
              })
              if (shouldFollowScrollRef.current) {
                setTimeout(() => scrollToBottom(), 30)
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          setMessages((prev) => {
            const newMsgs = [...prev]
            const last = newMsgs[newMsgs.length - 1]
            if (last && last.id === assistantMessage.id) {
              last.content = `错误: ${err.message}`
              last.isError = true
            }
            return newMsgs
          })
        }
      } finally {
        // 停止计时，盖章最终响应时间
        if (responseTimerRef.current) {
          clearInterval(responseTimerRef.current)
          responseTimerRef.current = null
        }
        if (responseStartRef.current) {
          const finalElapsed = (Date.now() - responseStartRef.current) / 1000
          assistantMessage.responseTime = parseFloat(finalElapsed.toFixed(1))
          setMessages((prev) => {
            const newMsgs = [...prev]
            const last = newMsgs[newMsgs.length - 1]
            if (last && last.id === assistantMessage.id) {
              last.responseTime = assistantMessage.responseTime
            }
            return newMsgs
          })
          responseStartRef.current = null
        }
        setLoading(false)
        isReceivingResponseRef.current = false
        abortRef.current = null
        activeAssistantIdRef.current = null
      }
    },
    [input, loading, scrollToBottom]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey) {
        // Shift+Enter：换行
        e.preventDefault()
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0
        const newValue = input.slice(0, start) + '\n' + input.slice(end)
        setInput(newValue)
        setTimeout(() => {
          if (target.setSelectionRange) {
            target.setSelectionRange(start + 1, start + 1)
          }
          if (inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
            adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
          }
        }, 0)
      } else if (e.key === 'Enter' && !e.shiftKey) {
        // Enter：提交
        e.preventDefault()
        handleSubmit(e as any)
      }
    },
    [input, handleSubmit, adjustTextareaHeight]
  )

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const pastedText = e.clipboardData.getData('text')
      if (pastedText.includes('\n')) {
        e.preventDefault()
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0
        const newValue = input.slice(0, start) + pastedText + input.slice(end)
        setInput(newValue)
        setTimeout(() => {
          if (inputRef.current && inputRef.current.setSelectionRange) {
            const newCursorPosition = start + pastedText.length
            inputRef.current.setSelectionRange(newCursorPosition, newCursorPosition)
          }
        }, 0)
      }
    },
    [input]
  )

  // 切换 Input/Textarea 时保持焦点和光标位置
  useEffect(() => {
    if (inputRef.current) {
      const currentElement = inputRef.current
      const cursorPosition = currentElement.selectionStart || input.length
      requestAnimationFrame(() => {
        currentElement.focus()
        if (currentElement.setSelectionRange) {
          currentElement.setSelectionRange(cursorPosition, cursorPosition)
        }
      })
    }
  }, [hasMultipleLines, input.length])

  // 切到多行模式时调整 textarea 高度
  useEffect(() => {
    if (hasMultipleLines && inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
      adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
    }
  }, [hasMultipleLines, input, adjustTextareaHeight])

  // 滚动监听：用户上滑时不强制拉回底部，回到底部恢复跟随（参考 RetrievalView）
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    const handleWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > 10 && !isFormInteractionRef.current) {
        shouldFollowScrollRef.current = false
      }
    }

    const handleScroll = throttle(() => {
      if (programmaticScrollRef.current) {
        programmaticScrollRef.current = false
        return
      }
      const c = messagesContainerRef.current
      if (c) {
        const isAtBottom = c.scrollHeight - c.scrollTop - c.clientHeight < 20
        if (isAtBottom) {
          shouldFollowScrollRef.current = true
        } else if (!isFormInteractionRef.current && !isReceivingResponseRef.current) {
          shouldFollowScrollRef.current = false
        }
      }
    }, 30)

    container.addEventListener('wheel', handleWheel as EventListener)
    container.addEventListener('scroll', handleScroll as EventListener)
    return () => {
      container.removeEventListener('wheel', handleWheel as EventListener)
      container.removeEventListener('scroll', handleScroll as EventListener)
    }
  }, [])

  // 表单交互时不关闭跟随（避免点输入框/按钮时 auto-scroll 被误关）
  useEffect(() => {
    const form = document.querySelector('form')
    if (!form) return
    const handleFormMouseDown = () => {
      isFormInteractionRef.current = true
      setTimeout(() => {
        isFormInteractionRef.current = false
      }, 500)
    }
    form.addEventListener('mousedown', handleFormMouseDown)
    return () => form.removeEventListener('mousedown', handleFormMouseDown)
  }, [])

  // 消息变化时跟随滚动（debounce，参考 RetrievalView）
  const debouncedMessages = useDebounce(messages, 150)
  useEffect(() => {
    if (shouldFollowScrollRef.current) {
      scrollToBottom()
    }
  }, [debouncedMessages, scrollToBottom])

  // 卸载清理：abort 在飞请求 + 清理定时器，防内存泄漏
  useEffect(() => {
    return () => {
      const controller = abortRef.current
      abortRef.current = null
      controller?.abort()
      if (responseTimerRef.current) {
        clearInterval(responseTimerRef.current)
        responseTimerRef.current = null
      }
      responseStartRef.current = null
      if (stopCooldownTimerRef.current) {
        clearTimeout(stopCooldownTimerRef.current)
        stopCooldownTimerRef.current = null
      }
    }
  }, [])

  // ── content 临时框操作 ──
  const addBox = useCallback(() => {
    setContents((prev) => [...prev, { key: nextKey, text: '', storedId: null }])
    setNextKey((k) => k + 1)
  }, [nextKey])

  const removeBox = useCallback((key: number) => {
    setContents((prev) => (prev.length <= 1 ? prev : prev.filter((c) => c.key !== key)))
  }, [])

  const updateBox = useCallback((key: number, text: string) => {
    setContents((prev) => prev.map((c) => (c.key === key ? { ...c, text } : c)))
  }, [])

  const storeContent = useCallback(
    async (key: number) => {
      const box = contents.find((c) => c.key === key)
      if (!box || !box.text) return
      try {
        const r = await fetch(`${AGENT_API}/api/content`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: box.text }),
        })
        const data = await r.json()
        if (data.content_id) {
          setContents((prev) =>
            prev.map((c) => (c.key === key ? { ...c, storedId: data.content_id } : c))
          )
          toast.success(`已暂存 #${data.content_id}`)
        }
      } catch (err: any) {
        toast.error(`暂存失败（dp_server 没起?）: ${err.message}`)
      }
    },
    [contents]
  )

  const handleCopyContent = useCallback(async (box: ContentBox) => {
    const text = box.storedId !== null ? `#${box.storedId}` : box.text
    if (!text.trim()) {
      toast.error('无内容可复制')
      return
    }
    const result = await copyToClipboard(text)
    if (result.success) {
      toast.success('已复制')
    } else {
      toast.error(result.error || '复制失败')
    }
  }, [])

  const handleCopyMessage = useCallback(async (message: MessageWithError) => {
    const contentToCopy = message.content || ''
    if (!contentToCopy.trim()) {
      toast.error('无内容可复制')
      return
    }
    const result = await copyToClipboard(contentToCopy)
    if (result.success) {
      toast.success('已复制')
    } else {
      toast.error(result.error || '复制失败')
    }
  }, [])

  // ── 清空 / 停止 ──
  const clearMessages = useCallback(() => {
    if (responseTimerRef.current) {
      clearInterval(responseTimerRef.current)
      responseTimerRef.current = null
    }
    responseStartRef.current = null
    setMessages([])
  }, [])

  const handleStop = useCallback(() => {
    const controller = abortRef.current
    if (!controller) return
    controller.abort()
    abortRef.current = null

    // 标记当前 assistant 消息为已终止，冻结响应时间
    const activeId = activeAssistantIdRef.current
    let stoppedResponseTime: number | null = null
    if (responseStartRef.current) {
      stoppedResponseTime = parseFloat(
        ((Date.now() - responseStartRef.current) / 1000).toFixed(1)
      )
    }
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== activeId) return m
        return {
          ...m,
          isAborted: true,
          responseTime: stoppedResponseTime ?? m.responseTime,
        }
      })
    )

    if (responseTimerRef.current) {
      clearInterval(responseTimerRef.current)
      responseTimerRef.current = null
    }
    responseStartRef.current = null
    setLoading(false)
    isReceivingResponseRef.current = false
    if (stopCooldownTimerRef.current) {
      clearTimeout(stopCooldownTimerRef.current)
      stopCooldownTimerRef.current = null
    }
    setStopDisabled(false)
  }, [])

  return (
    <div className="flex size-full px-2 pb-12 overflow-hidden">
      {/* 左侧：agent 聊天 */}
      <div className="flex flex-col gap-4 min-w-0" style={{ width: `${100 - rightPct}%` }}>
        <div className="relative grow">
          <div
            ref={messagesContainerRef}
            className="bg-primary-foreground/60 absolute inset-0 flex flex-col overflow-auto rounded-lg border p-2"
          >
            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {messages.length === 0 ? (
                <div className="text-muted-foreground flex h-full items-center justify-center text-center text-lg">
                  先在右侧暂存 content，然后输入“存一下 #1”
                  <br />
                  也可以直接问问题，如“查一下长鑫相关的”
                </div>
              ) : (
                messages.map((message, idx) => (
                  <div
                    key={message.id}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} items-end gap-2`}
                  >
                    {message.role === 'user' && (
                      <Button
                        onClick={() => handleCopyMessage(message)}
                        className="mb-2 size-6 shrink-0 rounded-md opacity-60 transition-opacity hover:opacity-100"
                        tooltip="复制"
                        variant="ghost"
                        size="icon"
                      >
                        <CopyIcon className="size-4" />
                      </Button>
                    )}
                    <ChatMessage
                      message={message}
                      isTabActive={isAgentTabActive}
                      isQuerying={
                        idx === messages.length - 1 &&
                        message.role === 'assistant' &&
                        loading
                      }
                    />
                    {message.role === 'assistant' && (
                      <Button
                        onClick={() => handleCopyMessage(message)}
                        className="mb-2 size-6 shrink-0 rounded-md opacity-60 transition-opacity hover:opacity-100"
                        tooltip="复制"
                        variant="ghost"
                        size="icon"
                      >
                        <CopyIcon className="size-4" />
                      </Button>
                    )}
                  </div>
                ))
              )}
              <div ref={messagesEndRef} className="pb-1" />
            </div>
          </div>
        </div>

        <form
          onSubmit={handleSubmit}
          className="flex shrink-0 items-center gap-2"
          autoComplete="on"
          method="post"
          action="#"
          role="search"
        >
          {/* Hidden submit button to ensure form meets HTML standards */}
          <input type="submit" style={{ display: 'none' }} tabIndex={-1} />
          <Button
            type="button"
            variant="outline"
            onClick={clearMessages}
            disabled={loading}
            size="sm"
          >
            <EraserIcon />
            清空
          </Button>
          <div className="relative flex-1">
            <label htmlFor="agent-input" className="sr-only">
              输入消息
            </label>
            {hasMultipleLines ? (
              <Textarea
                ref={inputRef as React.RefObject<HTMLTextAreaElement>}
                id="agent-input"
                autoComplete="on"
                className="max-h-[120px] min-h-[40px] w-full overflow-y-auto"
                value={input}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="如：存一下 #1，来源 https://douyin.com/xxx"
                disabled={loading}
                rows={1}
                style={{
                  resize: 'none',
                  height: 'auto',
                  minHeight: '40px',
                  maxHeight: '120px',
                }}
                onInput={(e: React.FormEvent<HTMLTextAreaElement>) => {
                  const target = e.target as HTMLTextAreaElement
                  requestAnimationFrame(() => {
                    target.style.height = 'auto'
                    target.style.height = Math.min(target.scrollHeight, 120) + 'px'
                  })
                }}
              />
            ) : (
              <Input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                id="agent-input"
                autoComplete="on"
                className="w-full"
                value={input}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="如：存一下 #1，来源 https://douyin.com/xxx"
                disabled={loading}
              />
            )}
          </div>
          {loading ? (
            <Button
              type="button"
              variant="destructive"
              onClick={handleStop}
              disabled={stopDisabled}
              size="sm"
            >
              <SquareIcon />
              停止
            </Button>
          ) : (
            <Button type="submit" variant="default" size="sm" disabled={!input.trim()}>
              <SendIcon />
              发送
            </Button>
          )}
        </form>
      </div>

      {/* 拖拽分隔 */}
      <div
        onMouseDown={startDrag}
        className="w-1 shrink-0 cursor-col-resize bg-border/60 transition-colors hover:bg-primary/40"
      />

      {/* 右侧：content 临时框列表 */}
      <div className="flex min-w-0 flex-col" style={{ width: `${rightPct}%` }}>
        <Card className="flex h-full w-full shrink-0 flex-col">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 px-4 pb-2 pt-4">
            <CardTitle className="text-sm">Content 临时框</CardTitle>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={addBox}
              tooltip="增加框"
              className="size-7"
            >
              <PlusIcon className="size-4" />
            </Button>
          </CardHeader>
          <CardContent className="m-0 flex grow flex-col p-0">
            <div className="relative size-full">
              <div className="absolute inset-0 flex flex-col gap-2 overflow-auto p-2">
                {contents.map((box) => (
                  <div
                    key={box.key}
                    className="rounded-md border border-border bg-background p-2"
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-xs font-bold text-muted-foreground">
                        {box.storedId !== null ? `#${box.storedId}` : `框 ${box.key}（未暂存）`}
                      </span>
                      <div className="flex gap-0.5">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          onClick={() => handleCopyContent(box)}
                          tooltip={box.storedId !== null ? '复制编号' : '复制内容'}
                          className="size-6"
                        >
                          <CopyIcon className="size-3" />
                        </Button>
                        {contents.length > 1 && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => removeBox(box.key)}
                            tooltip="删除框"
                            className="size-6 hover:text-destructive"
                          >
                            <TrashIcon className="size-3" />
                          </Button>
                        )}
                      </div>
                    </div>
                    <Textarea
                      value={box.text}
                      onChange={(e) => updateBox(box.key, e.target.value)}
                      placeholder="粘贴 content 全文..."
                      className="h-32 w-full text-xs"
                    />
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      onClick={() => storeContent(box.key)}
                      disabled={!box.text}
                      className="mt-1 w-full"
                    >
                      {box.storedId !== null ? `已暂存 #${box.storedId}（重新暂存）` : '暂存'}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
