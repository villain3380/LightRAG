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

// deepseekProvider 读 DEEPSEEK_API_KEY，.env 用 LLM_BINDING_API_KEY，映射
process.env.DEEPSEEK_API_KEY =
  process.env.LLM_BINDING_API_KEY || process.env.DEEPSEEK_API_KEY || "";

const DATA_PLATFORM = process.env.DATA_PLATFORM_URL || "http://localhost:9955";

// === content 临时框 store（内存，编号 -> content 全文） ===
const contentStore = new Map<number, string>();
let contentIdCounter = 0;

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

// === system prompt ===
const SYSTEM_PROMPT = `你是 woowoo 的中台知识库管家。

工作方式：
- 用户在前端"content 临时框"粘贴 content 全文，暂存后得到编号（如 #1）
- 用户在聊天框告诉你"存一下 #1"或"把 #1 存入知识库"
- 你先调用 read_content(content_id=1) 读取 content 全文，理解内容
- 生成入库字段：title(≤30字), summary(概括~100-200字), iv_grade(S2/S1/A/B/C/D), iv_desc, domain, tags, source_url
- 调用 ingest_insight(title, summary, iv_grade, domain, tags, content_id=1, source_url) 入库
  重要：ingest_insight 传 content_id（编号），不要传 content 全文。工具会自动读 content 临时框。

iv_grade：用户主动给的默认 S2。
domain：financial_market / financial_market/semiconductor / technology / policy / daily_life / other
tags：3-7 个标签。

如果用户问问题（不是入库），用 search_insight 检索；要细节用 get_insight_content。

关键：ingest_insight 的 content_id 参数传编号，绝对不要传 content 全文。read_content 用来读 content 生成 summary。`;

// === 建 Agent（单 Agent，多轮对话状态共享，先单用户） ===
const models = createModels();
models.setProvider(deepseekProvider());
const model = models.getModel("deepseek", "deepseek-v4-flash");
if (!model) throw new Error("deepseek-v4-flash not found");

const agent = new Agent({
  initialState: {
    systemPrompt: SYSTEM_PROMPT,
    model,
    tools: [readContentTool, ingestInsightTool, searchInsightTool, getContentTool],
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

// 暂存 content，返回编号
app.post("/api/content", async (c) => {
  const { content } = await c.req.json<{ content: string }>();
  if (!content) return c.json({ error: "content required" }, 400);
  const id = ++contentIdCounter;
  contentStore.set(id, content);
  console.log(`content #${id} stored (${content.length} chars)`);
  return c.json({ content_id: id });
});

// agent 对话（非流式，先简单；后续可改 SSE 流式）
app.post("/api/chat", async (c) => {
  const { message } = await c.req.json<{ message: string }>();
  if (!message) return c.json({ error: "message required" }, 400);
  let response = "";
  const unsub = agent.subscribe((event: any) => {
    if (
      event.type === "message_update" &&
      event.assistantMessageEvent?.type === "text_delta"
    ) {
      response += event.assistantMessageEvent.delta;
    }
  });
  try {
    await agent.prompt(message);
  } catch (e: any) {
    response += `\n[error: ${e.message}]`;
  }
  unsub();
  return c.json({ response });
});

serve({ fetch: app.fetch, port: 9956 }, (info) => {
  console.log(`agent HTTP server: http://localhost:${info.port}`);
  console.log(`data_platform: ${DATA_PLATFORM}`);
  console.log("endpoints: POST /api/content, POST /api/chat, GET /api/health");
});
