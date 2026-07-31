/**
 * 中台知识库管家 agent（CLI）。
 *
 * LLM: deepseek-v4-flash（OpenAI 兼容，pi-ai deepseekProvider）
 * 工具: ingest_insight / search_insight / get_insight_content
 *       （execute 内部 fetch HTTP 调 data_platform FastAPI :9700）
 * 预览确认: beforeToolCall 钩子拦截 ingest_insight，打印参数，stdin 读 y/n
 *
 * 跑法（先起 data_platform: uvicorn data_platform.main:app --port 9700）:
 *   cd agent
 *   node --env-file=../.env --import tsx dp_agent.ts
 *
 * .env 需要: LLM_BINDING_API_KEY（deepseek key，代码里映射到 DEEPSEEK_API_KEY）
 */
import { Agent } from "./src/index.ts";
import { createModels } from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
import { Type } from "typebox";
import * as readline from "node:readline/promises";

// deepseekProvider 读 DEEPSEEK_API_KEY，.env 用 LLM_BINDING_API_KEY，映射
process.env.DEEPSEEK_API_KEY =
  process.env.LLM_BINDING_API_KEY || process.env.DEEPSEEK_API_KEY || "";

const DATA_PLATFORM = process.env.DATA_PLATFORM_URL || "http://localhost:9955";

// === 工具：HTTP 调 data_platform FastAPI ===
const ingestInsightTool = {
  name: "ingest_insight",
  label: "入库 insight",
  description:
    "把一条高价值信息入库（PG + Milvus）。调用前会先让用户确认参数。" +
    "参数：title(短标题), summary(概括), content或content_ref(原文/文件引用), " +
    "iv_grade(S2/S1/A/B/C/D), iv_desc(等级原因), domain(领域), tags(标签数组), " +
    "source_type, source_url, source_title",
  parameters: Type.Object({
    title: Type.String({ description: "短标题 ≤30 字" }),
    summary: Type.String({ description: "概括 ~100-200 字" }),
    content: Type.Optional(Type.String({ description: "原文（和 content_ref 二选一）" })),
    content_ref: Type.Optional(Type.String({ description: "文件引用（和 content 二选一）" })),
    iv_grade: Type.String({ description: "价值等级 S2/S1/A/B/C/D" }),
    iv_desc: Type.Optional(Type.String({ description: "等级原因" })),
    domain: Type.String({
      description: "知识领域（financial_market / financial_market/semiconductor / technology / policy / daily_life / frontend_and_backend / other）",
    }),
    tags: Type.Optional(Type.Array(Type.String(), { description: "标签数组" })),
    source_type: Type.Optional(Type.String()),
    source_url: Type.Optional(Type.String()),
    source_title: Type.Optional(Type.String()),
  }),
  execute: async (_id: string, params: any) => {
    const r = await fetch(`${DATA_PLATFORM}/ingest/insight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(`ingest failed: ${JSON.stringify(result)}`);
    return {
      content: [{ type: "text" as const, text: `入库成功：${JSON.stringify(result)}` }],
      details: result,
    };
  },
};

const searchInsightTool = {
  name: "search_insight",
  label: "检索 insight",
  description:
    "语义检索 insight（hybrid dense+sparse），返回摘要列表（不含原文）。参数：query, domain(可选), top_k(默认5)",
  parameters: Type.Object({
    query: Type.String({ description: "查询文本" }),
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

const getContentTool = {
  name: "get_insight_content",
  label: "读 insight 原文",
  description: "按 id 读 insight 原文（深入时调）。参数：id",
  parameters: Type.Object({ id: Type.Number({ description: "insight id" }) }),
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
const SYSTEM_PROMPT = `你是 woowoo 的中台知识库管家。用户会给你高价值信息文本（可能附带来源链接）。

你的职责：
1. 理解文本，提取/生成入库字段：
   - title：短标题 ≤30 字
   - summary：概括 ~100-200 字（精炼准确）
   - iv_grade：价值等级 S2/S1/A/B/C/D（用户主动给的默认 S2）
   - iv_desc：等级原因
   - domain：知识领域（financial_market / financial_market/semiconductor / technology / policy / daily_life / frontend_and_backend / other）
   - tags：3-7 个标签
   - source_url：来源链接（如果用户给了）
2. 调用 ingest_insight 工具入库（调用前系统会让你确认参数）
3. 如果用户问问题（不是入库），用 search_insight 检索，返回摘要；用户要细节再 get_insight_content

重要：调 ingest_insight 前，系统会打印参数让用户确认。你先生成完整参数，再调工具。

用户给的文本就是完整 content，直接用作 content 字段，不要追问用户补充原文。

生成字段后，**必须立即调用 ingest_insight 工具入库**。不要用文本输出"请确认"再等用户回话--系统会自动弹参数让用户确认（y/n）。你只管直接调工具，把字段作为参数传给 ingest_insight。禁止用文字问"是否确认入库"。`;

// === 建 Agent ===
const models = createModels();
models.setProvider(deepseekProvider());
const model = models.getModel("deepseek", "deepseek-v4-flash");
if (!model) throw new Error("deepseek-v4-flash not found in deepseek provider");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

const agent = new Agent({
  initialState: {
    systemPrompt: SYSTEM_PROMPT,
    model,
    tools: [ingestInsightTool, searchInsightTool, getContentTool],
  },
  streamFunction: models.streamSimple.bind(models),
  // 钩子：ingest_insight 前确认
  beforeToolCall: async ({ toolCall, args }: any) => {
    if (toolCall.name === "ingest_insight") {
      console.log("\n=== 即将入库 insight，参数预览 ===");
      console.log(JSON.stringify(args, null, 2));
      if (process.env.DP_AUTO_CONFIRM === "y") {
        console.log("（DP_AUTO_CONFIRM=y，自动确认）");
      } else {
        const ans = await rl.question("确认入库？(y/n): ");
        if (ans.trim().toLowerCase() !== "y") {
          return { block: true, reason: "用户取消入库" };
        }
      }
    }
    return undefined;
  },
});

// 流式输出 LLM 文本
agent.subscribe((event: any) => {
  if (
    event.type === "message_update" &&
    event.assistantMessageEvent?.type === "text_delta"
  ) {
    process.stdout.write(event.assistantMessageEvent.delta);
  }
});

// === CLI 主循环 ===
async function main() {
  console.log("中台知识库管家已就绪（deepseek-v4-flash）。");
  console.log("输入文本入库，或问问题检索。Ctrl+C 退出。\n");
  while (true) {
    const input = await rl.question("你> ");
    if (!input.trim()) continue;
    process.stdout.write("管家> ");
    await agent.prompt(input);
    process.stdout.write("\n");
  }
}

main().catch((e) => {
  console.error("agent error:", e);
  process.exit(1);
});
