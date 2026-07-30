/**
 * 中台 agent HTTP server（Hono）。
 *
 * content 临时框（编号 + 内存 store）+ agent 多轮对话。
 * agent 工具：read_content / ingest_insight(content_id) / search_insight / get_insight_content
 * agent 不输出 content（ingest_insight 传 content_id，工具读 store 入库），省 output token。
 *
 * 跑法（先起 data_platform :9955）:
 *   cd agent
 *   node --env-file=../.env --import tsx dp_server.ts
 *
 * 端口：9956（data_platform 9955）
 */
import { Agent } from "./src/index.ts";
import { createModels } from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
import { Type } from "typebox";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { streamSSE } from "hono/streaming";
import pg from "pg";
import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

// deepseekProvider 读 DEEPSEEK_API_KEY，.env 用 LLM_BINDING_API_KEY，映射
process.env.DEEPSEEK_API_KEY =
  process.env.LLM_BINDING_API_KEY || process.env.DEEPSEEK_API_KEY || "";

const DATA_PLATFORM = process.env.DATA_PLATFORM_URL || "http://localhost:9955";
const LIGHTRAG_API = process.env.LIGHTRAG_API_URL || "http://localhost:9621";

// === content 临时框 store（内存，编号 -> content 全文） ===
const contentStore = new Map<number, string>();
let contentIdCounter = 0;

// === 检索路径追踪 ALS（把 original_query/turn_id/rewrite_ms 透传给 /query） ===
// 每次 /api/chat 的 agent.prompt 包在 ALS.run 里；ragQueryTool.execute() 读出
// ctx 算 rewrite_ms（= agent 思考到工具调用的延迟），生成 trace_id，连同
// turn_id/call_index 一起 POST 给 /query，lightrag 落 shared.query_trace。
type QueryTraceCtx = {
  originalQuery: string;
  startTime: number;
  turnId: string;
  callIndex: number;
};
const queryTraceALS = new AsyncLocalStorage<QueryTraceCtx>();

// === agent 工具 ===

/** 读 content 临时框（agent 看 content 生成 summary，content 进 input 不进 output） */
const readContentTool = {
  name: "read_content",
  label: "读 content",
  description:
    "读取 content 临时框的内容（按编号）。用户暂存的 content 全文在这里。生成 summary 前先 read_content 看 content。",
  parameters: Type.Object({
    content_id: Type.Number({ description: "content 临时框编号" }),
  }),
  execute: async (_id: string, params: any) => {
    const content = contentStore.get(params.content_id);
    if (!content) throw new Error(`content #${params.content_id} 不存在`);
    return {
      content: [{ type: "text" as const, text: content }],
      details: { content_id: params.content_id, length: content.length },
    };
  },
};

/** 入库 insight：传 content_id（不是 content 全文），工具读 store 入库 data_platform */
const ingestInsightTool = {
  name: "ingest_insight",
  label: "入库 insight",
  description:
    "入库 insight。传 content_id（不是 content 全文），工具自动读 content 临时框。" +
    "参数：title, summary, iv_grade(S2/S1/A/B/C/D), iv_desc, domain, tags, content_id, source_type, source_url, source_title",
  parameters: Type.Object({
    title: Type.String({ description: "短标题 ≤30 字" }),
    summary: Type.String({ description: "概括 ~100-200 字" }),
    iv_grade: Type.String({ description: "S2/S1/A/B/C/D" }),
    iv_desc: Type.Optional(Type.String()),
    domain: Type.String({
      description: "financial_market / financial_market/semiconductor / technology / policy / daily_life / other",
    }),
    tags: Type.Optional(Type.Array(Type.String())),
    content_id: Type.Number({ description: "content 临时框编号（不是 content 全文！）" }),
    source_type: Type.Optional(Type.String()),
    source_url: Type.Optional(Type.String()),
    source_title: Type.Optional(Type.String()),
  }),
  execute: async (_id: string, params: any) => {
    const content = contentStore.get(params.content_id);
    if (!content) throw new Error(`content #${params.content_id} 不存在`);
    const { content_id, ...fields } = params;
    const r = await fetch(`${DATA_PLATFORM}/ingest/insight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...fields, content }),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`ingest failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `入库成功：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

/** 检索 insight（代理 data_platform） */
const searchInsightTool = {
  name: "search_insight",
  label: "检索 insight",
  description: "语义检索 insight（hybrid dense+sparse），返回摘要列表（不含原文）。参数：query, domain(可选), top_k(默认5)",
  parameters: Type.Object({
    query: Type.String(),
    domain: Type.Optional(Type.String()),
    top_k: Type.Optional(Type.Number()),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/insight/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`search failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
      details: { count: Array.isArray(result) ? result.length : 0 },
    };
  },
};

/** 读 insight 原文（代理 data_platform） */
const getContentTool = {
  name: "get_insight_content",
  label: "读 insight 原文",
  description: "按 id 读 insight 原文（深入时调）。参数：id",
  parameters: Type.Object({ id: Type.Number() }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/insight/${params.id}/content`);
    const result = await r.json();
    if (!r.ok) throw new Error(`get content failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result) }],
      details: { id: params.id },
    };
  },
};

/** 批量更新 insight */
const updateInsightTool = {
  name: "update_insight",
  label: "更新 insight",
  description:
    "批量更新 insight。where 条件（created_after: YYYY-MM-DD, created_before, domain, iv_grade）+ set 字段（source_title, source_url, source_type, iv_grade, iv_desc, domain, title, summary）。" +
    "例如：把今天入库的 insight 的 source_title 改成'大锅饭评房记'，where={created_after:'2026-07-27'}, set={source_title:'大锅饭评房记'}",
  parameters: Type.Object({
    where: Type.Object({
      id: Type.Optional(Type.Number({ description: "按 id 精确匹配单个" })),
      ids: Type.Optional(Type.Array(Type.Number(), { description: "按 id 列表匹配多个" })),
      title_contains: Type.Optional(Type.String({ description: "标题模糊匹配（ILIKE）" })),
      created_after: Type.Optional(Type.String({ description: "YYYY-MM-DD，查此日期之后入库的" })),
      created_before: Type.Optional(Type.String({ description: "YYYY-MM-DD" })),
      domain: Type.Optional(Type.String()),
      iv_grade: Type.Optional(Type.String()),
    }),
    set: Type.Object({
      source_title: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      source_url: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      source_type: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      iv_grade: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      iv_desc: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      domain: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      title: Type.Optional(Type.Union([Type.String(), Type.Null()])),
      summary: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    }),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/insight/update`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`update failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `更新完成：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

/** 检索 lightrag 知识库（文档切片 + 知识图谱 + 向量 hybrid） */
const ragQueryTool = {
  name: "rag_query",
  label: "RAG 检索",
  description:
    "检索 lightrag 知识库（文档切片 + 知识图谱 + 向量 hybrid）。" +
    "mode：naive(纯向量快) / local(局部图) / global(全局图) / hybrid(图+向量) / mix(全覆盖,最全) / bypass(跳过检索)。" +
    "简单事实用 naive，复杂关联用 hybrid/mix。" +
    "用户可能在消息里指定参数（'用户要求用以下参数检索：...'），按指定参数调。",
  parameters: Type.Object({
    query: Type.String({ description: "查询文本" }),
    mode: Type.Optional(Type.String({ description: "检索模式：naive/local/global/hybrid/mix/bypass，默认 mix" })),
    top_k: Type.Optional(Type.Number({ description: "实体/关系检索数，默认 40" })),
    chunk_top_k: Type.Optional(Type.Number({ description: "chunk 检索数，默认 20" })),
    dense_weight: Type.Optional(Type.Number({ description: "dense 权重 0-1，默认 0.5" })),
    sparse_weight: Type.Optional(Type.Number({ description: "sparse 权重 0-1，默认 0.5" })),
  }),
  execute: async (_id: string, params: any) => {
    // 从 ALS 取本次 turn 的追踪上下文，算 rewrite_ms + 生成 trace_id 透传给 /query
    const ctx = queryTraceALS.getStore();
    const traceFields: Record<string, unknown> = {};
    if (ctx) {
      traceFields.trace_id = `qt_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
      traceFields.turn_id = ctx.turnId;
      traceFields.call_index = ++ctx.callIndex;
      traceFields.rewrite_ms = Date.now() - ctx.startTime;
      traceFields.source = "dp_server";
      traceFields.original_query = ctx.originalQuery;
    }
    const r = await fetch(`${LIGHTRAG_API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: params.query,
        mode: params.mode || "mix",
        top_k: params.top_k,
        chunk_top_k: params.chunk_top_k,
        dense_weight: params.dense_weight,
        sparse_weight: params.sparse_weight,
        stream: false,
        response_type: "Multiple Paragraphs",
        ...traceFields,
      }),
    });
    if (!r.ok) throw new Error(`rag_query failed: ${r.status} ${await r.text()}`);
    const result = await r.json();
    return {
      content: [{ type: "text" as const, text: result.response || JSON.stringify(result) }],
      details: { references: result.references },
    };
  },
};

// === todo 工具（代理 data_platform /todo/） ===

/** 新建 todo */
const todoCreateTool = {
  name: "todo_create",
  label: "新建待办",
  description: "新建一条待办事项。priority: P0/P1/P2/P3(默认P2), due_date: YYYY-MM-DD。返回 {id}。",
  parameters: Type.Object({
    title: Type.String({ description: "标题" }),
    priority: Type.Optional(Type.String({ description: "P0/P1/P2/P3, 默认 P2" })),
    detail: Type.Optional(Type.String({ description: "详细描述" })),
    due_date: Type.Optional(Type.String({ description: "截止日期 YYYY-MM-DD" })),
    tags: Type.Optional(Type.Array(Type.String())),
    domain: Type.Optional(Type.String()),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/todo/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`create todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `已新建待办：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

/** 列出 todo */
const todoListTool = {
  name: "todo_list",
  label: "列出待办",
  description: "列出待办事项(按 priority P0->P3, due_date 排序)。可按 status(todo/in_progress/done/cancelled)/priority/domain 筛选。默认返回全部状态。",
  parameters: Type.Object({
    status: Type.Optional(Type.String({ description: "todo/in_progress/done/cancelled" })),
    priority: Type.Optional(Type.String()),
    domain: Type.Optional(Type.String()),
    limit: Type.Optional(Type.Number()),
  }),
  execute: async (_id: string, params: any) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v != null) qs.set(k, String(v));
    const r = await fetch(`${DATA_PLATFORM}/todo/?${qs}`);
    const result = await r.json();
    if (!r.ok) throw new Error(`list todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
      details: { count: Array.isArray(result) ? result.length : 0 },
    };
  },
};

/** 取单条 todo */
const todoGetTool = {
  name: "todo_get",
  label: "查看待办",
  description: "按 id 取单条待办(含 detail)。",
  parameters: Type.Object({ id: Type.Number() }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/todo/${params.id}`);
    const result = await r.json();
    if (!r.ok) throw new Error(`get todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result) }],
      details: { id: params.id },
    };
  },
};

/** 更新 todo */
const todoUpdateTool = {
  name: "todo_update",
  label: "更新待办",
  description: "更新待办。set: {title/detail/status/priority/due_date/tags/domain/sort_order}。status='done'自动填completed_at;status='cancelled'软删除。",
  parameters: Type.Object({
    id: Type.Number(),
    set: Type.Object({
      title: Type.Optional(Type.String()),
      detail: Type.Optional(Type.String()),
      status: Type.Optional(Type.String({ description: "todo/in_progress/done/cancelled" })),
      priority: Type.Optional(Type.String()),
      due_date: Type.Optional(Type.String({ description: "YYYY-MM-DD" })),
      tags: Type.Optional(Type.Array(Type.String())),
      domain: Type.Optional(Type.String()),
      sort_order: Type.Optional(Type.Number()),
    }),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/todo/${params.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ set: params.set }),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`update todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `已更新：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

/** 标记完成 */
const todoCompleteTool = {
  name: "todo_complete",
  label: "完成待办",
  description: "按 id 标记待办完成(trigger 自动填 completed_at)。",
  parameters: Type.Object({ id: Type.Number() }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/todo/${params.id}/complete`, { method: "PUT" });
    const result = await r.json();
    if (!r.ok) throw new Error(`complete todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `已完成：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

/** 语义搜索 todo（模糊复述找待办） */
const todoSearchTool = {
  name: "todo_search",
  label: "搜索待办",
  description:
    "语义搜索 todo（用模糊复述找某条待办）。例如用户说'我上周说的测试重排模型延迟'，用此工具找到对应 todo。返回按相似度排序的列表（含 score/created_at/status）。参数：query, top_k(默认5)",
  parameters: Type.Object({
    query: Type.String({ description: "查询文本（用户的复述）" }),
    top_k: Type.Optional(Type.Number()),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/todo/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: params.query, top_k: params.top_k ?? 5 }),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`search todo failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
      details: { count: Array.isArray(result) ? result.length : 0 },
    };
  },
};

// === system prompt ===
const SYSTEM_PROMPT = `你是 woowoo 的中台知识库管家。

工作方式：
- 用户在前端"content 临时框"粘贴 content 全文，暂存后得到编号（如 #1）
- 用户在聊天框告诉你"存一下 #1"或"把 #1 存入知识库"
- 你先调用 read_content(content_id=1) 读取 content 全文，理解内容
- 生成入库字段：title(≤30字), summary(概括~100-200字), iv_grade(S2/S1/A/B/C/D), iv_desc, domain, tags, source_url
- 调用 ingest_insight(title, summary, iv_grade, domain, tags, content_id=1, source_url) 入库
  重要：ingest_insight 传 content_id（编号），不要传 content 全文。工具会自动读 content 临时框。

iv_grade：用户主动给的默认 S1。
domain：financial_market / financial_market/semiconductor / technology / policy / daily_life / other
tags：3-7 个标签。

如果用户问问题（不是入库），用 search_insight 检索；要细节用 get_insight_content。

关键：ingest_insight 的 content_id 参数传编号，绝对不要传 content 全文。read_content 用来读 content 生成 summary。

知识库检索（rag_query）：
- 用户问知识库内容（文档/研报/资料，不是 insight）时，用 rag_query 检索 lightrag 知识库。
- 调用时把用户问题改写成更可能检索到正确结果的 query：去掉口语化措辞，补上关键实体/术语，用文档里可能出现的表达。例如"那个重排模型延迟测了没"改写成"重排模型 推理延迟 测试结果"。
- 简单事实用 naive；复杂关联用 hybrid/mix。为提高一次召回成功率，可一次发多个 rag_query（不同角度的 query + 不同 top_k）。

待办事项（todo）：
- 用户说"记一下/提醒我/待办"时，用 todo_create(title, priority, due_date, detail) 新建。priority 默认 P2，紧急用 P0/P1。
- 用户问"我有什么待办/没完成的"时，用 todo_list(status='todo') 列出未完成。
- 用户说"完成了/做完了第N条"时，用 todo_complete(id) 标记完成。
- 改待办内容/优先级/截止日用 todo_update(id, set)。
- 用户用模糊复述找某条待办时（如"我上周说的测试重排模型延迟完成了"），先用 todo_search(query) 找到对应 todo，看 created_at/status 确认是哪条，再 todo_complete 或 todo_update。`;

// === 建 Agent（单 Agent，多轮对话状态共享，先单用户） ===
const models = createModels();
models.setProvider(deepseekProvider());
const model = models.getModel("deepseek", "deepseek-v4-flash");
if (!model) throw new Error("deepseek-v4-flash not found");

const agent = new Agent({
  initialState: {
    systemPrompt: SYSTEM_PROMPT,
    model,
    tools: [readContentTool, ingestInsightTool, searchInsightTool, getContentTool, updateInsightTool, ragQueryTool, todoCreateTool, todoListTool, todoGetTool, todoUpdateTool, todoCompleteTool, todoSearchTool],
  },
  streamFunction: models.streamSimple.bind(models),
});

// === Hono HTTP server ===
const app = new Hono();

// CORS：允许 lightrag_webui（dev 5173/5174，prod 4173）
app.use(
  "*",
  cors({
    origin: ["http://localhost:5173", "http://localhost:5174", "http://localhost:4173"],
    allowMethods: ["GET", "POST"],
    allowHeaders: ["Content-Type"],
  }),
);

app.get("/api/health", (c) => c.json({ status: "ok" }));

// === PG 连接（agent_session 持久化） ===
const pgPool = new pg.Pool({
  host: "localhost",
  port: 5432,
  user: "agent_session_writer",
  password: process.env.AGENT_SESSION_PWD || "asw_2026",
  database: "rag",
});

// === 会话端点 ===
// 新建会话
app.post("/api/sessions", async (c) => {
  const { title } = await c.req.json().catch(() => ({}));
  const r = await pgPool.query(
    "INSERT INTO agent_session.session(title) VALUES($1) RETURNING id, title, created_at",
    [title || null]
  );
  return c.json(r.rows[0]);
});

// 会话列表
app.get("/api/sessions", async (c) => {
  const r = await pgPool.query(
    "SELECT id, title, created_at, updated_at FROM agent_session.session ORDER BY updated_at DESC LIMIT 50"
  );
  return c.json(r.rows);
});

// 恢复会话消息
app.get("/api/sessions/:id", async (c) => {
  const id = c.req.param("id");
  const r = await pgPool.query(
    "SELECT id, session_id, role, content, display_content, tool_name, tool_args, tool_result, response_time FROM agent_session.message WHERE session_id=$1 ORDER BY created_at",
    [id]
  );
  return c.json(r.rows);
});

// 删除会话
app.delete("/api/sessions/:id", async (c) => {
  const id = c.req.param("id");
  await pgPool.query("DELETE FROM agent_session.session WHERE id=$1", [id]);
  return c.json({ status: "ok" });
});

// 暂存 content，返回编号
app.post("/api/content", async (c) => {
  const { content } = await c.req.json<{ content: string }>();
  if (!content) return c.json({ error: "content required" }, 400);
  const id = ++contentIdCounter;
  contentStore.set(id, content);
  console.log(`content #${id} stored (${content.length} chars)`);
  return c.json({ content_id: id });
});

// agent 对话（SSE 流式 + 持久化到 agent_session）
app.post("/api/chat", async (c) => {
  const { message, session_id } = await c.req.json<{ message: string; session_id?: number }>();
  if (!message) return c.json({ error: "message required" }, 400);
  if (!session_id) return c.json({ error: "session_id required" }, 400);

  // 存 user 消息
  await pgPool.query(
    "INSERT INTO agent_session.message(session_id, role, content) VALUES($1, 'user', $2)",
    [session_id, message]
  );

  let agentText = "";

  return streamSSE(c, async (stream) => {
    const toolCallInfo = new Map<string, { name: string; args: any }>();
    const unsub = agent.subscribe(async (event: any) => {
      if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
        agentText += event.assistantMessageEvent.delta;
        await stream.writeSSE({ data: JSON.stringify({ type: "text", delta: event.assistantMessageEvent.delta }) });
      } else if (event.type === "tool_execution_start") {
        // 工具前文本存 assistant 消息（如果有）
        if (agentText.trim()) {
          await pgPool.query(
            "INSERT INTO agent_session.message(session_id, role, content, display_content) VALUES($1, 'assistant', $2, $3)",
            [session_id, agentText, agentText]
          );
          agentText = "";
        }
        toolCallInfo.set(event.toolCallId, { name: event.toolName, args: event.args });
        await stream.writeSSE({ data: JSON.stringify({ type: "tool_call", id: event.toolCallId, name: event.toolName, args: event.args }) });
      } else if (event.type === "tool_execution_end") {
        const result = event.result?.details ?? event.result;
        const info = toolCallInfo.get(event.toolCallId);
        // 存 tool_call 消息（name + args 从 start 事件取，end 事件只有 result）
        await pgPool.query(
          "INSERT INTO agent_session.message(session_id, role, tool_name, tool_args, tool_result) VALUES($1, 'tool_call', $2, $3, $4)",
          [session_id, info?.name ?? event.toolName, JSON.stringify(info?.args), JSON.stringify(result)]
        );
        await stream.writeSSE({ data: JSON.stringify({ type: "tool_result", id: event.toolCallId, name: info?.name ?? event.toolName, result }) });
      }
    });
    try {
      // 包在 ALS.run 里：ragQueryTool.execute() 能读到 originalQuery/startTime/
      // turnId，算 rewrite_ms + 生成 trace_id 透传给 /query（lightrag 落 query_trace）
      await queryTraceALS.run(
        {
          originalQuery: message,
          startTime: Date.now(),
          turnId: `turn_${randomUUID().replace(/-/g, "").slice(0, 12)}`,
          callIndex: 0,
        },
        () => agent.prompt(message),
      );
    } catch (e: any) {
      await stream.writeSSE({ data: JSON.stringify({ type: "error", message: e.message }) });
    }
    unsub();
    // 存剩余 agentText（工具后文本）
    if (agentText.trim()) {
      await pgPool.query(
        "INSERT INTO agent_session.message(session_id, role, content, display_content) VALUES($1, 'assistant', $2, $3)",
        [session_id, agentText, agentText]
      );
    }
    // 更新 session.updated_at + title（首条消息摘要）
    await pgPool.query(
      "UPDATE agent_session.session SET updated_at=now(), title=COALESCE(NULLIF(title, ''), $2) WHERE id=$1",
      [session_id, message.slice(0, 50)]
    );
    await stream.writeSSE({ data: JSON.stringify({ type: "done" }) });
  });
});

serve({ fetch: app.fetch, port: 9956 }, (info) => {
  console.log(`agent HTTP server: http://localhost:${info.port}`);
  console.log(`data_platform: ${DATA_PLATFORM}`);
  console.log("endpoints: POST /api/content, POST /api/chat, GET /api/health");
});
