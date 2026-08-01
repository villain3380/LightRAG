/**
 * 记忆层 compact：上下文超阈值时，把旧消息 LLM 总结成 summary，保留最近 keepRecentTokens。
 * 复用 pi-ai SDK 的 estimateContextTokens / shouldCompact / generateSummaryWithUsage。
 *
 * 触发：shouldCompact(tokens, contextWindow, settings) = tokens > contextWindow - reserveTokens。
 * deepseek-v4-flash contextWindow=1M，reserve=16384 -> 约 983k 触发；keepRecent=200k 保留。
 * 摘要支持迭代更新（已有 summary 时用 UPDATE prompt，不从头总结）。
 *
 * 落盘：summary + first_kept_message_id 写 agent_session.compaction；_state.messages 裁成
 * [新 compactionSummary] + [recent]。冷启动按 compaction 表 + message_json 还原。
 */
import type { Agent } from "./src/agent.ts";
import type { AgentMessage } from "./src/types.ts";
import type { Model, Models } from "@earendil-works/pi-ai";
import {
	estimateContextTokens,
	estimateTokens,
	generateSummaryWithUsage,
	shouldCompact,
} from "./src/harness/compaction/compaction.ts";
import { createCompactionSummaryMessage } from "./src/harness/messages.ts";
import { getPgId, persistCompaction } from "./session-store.ts";

const KEEP_RECENT = parseInt(process.env.KEEP_RECENT_TOKENS || "200000", 10);
const RESERVE = parseInt(process.env.RESERVE_TOKENS || "16384", 10);
const CONTEXT_WINDOW_FALLBACK = parseInt(process.env.AGENT_CONTEXT_WINDOW || "1000000", 10);
const SETTINGS = { enabled: true, reserveTokens: RESERVE, keepRecentTokens: KEEP_RECENT };

/** 找切点：保留最近 keepRecentTokens，尽量在 user 消息边界切（不切断一轮）。 */
function findCutIndex(messages: AgentMessage[], keepRecentTokens: number): number {
	const start = messages[0]?.role === "compactionSummary" ? 1 : 0;
	let acc = 0;
	let candidate = messages.length;
	for (let i = messages.length - 1; i >= start; i--) {
		acc += estimateTokens(messages[i]);
		if (acc >= keepRecentTokens) {
			candidate = i;
			break;
		}
	}
	// 往回找到 user 消息边界（一轮的起点），避免切到 assistant+toolResult 中间
	for (let i = candidate; i > start; i--) {
		if (messages[i].role === "user") return i;
	}
	return candidate;
}

/** 上下文超阈值时 compact。turn 前调（不在 loop 里嵌套 LLM 调用）。失败只 warn，不阻断。 */
export async function maybeCompact(
	agent: Agent,
	pool: any,
	sessionId: number,
	models: Models,
	model: Model<any>,
): Promise<void> {
	const messages = agent.state.messages;
	const tokens = estimateContextTokens(messages).tokens;
	const contextWindow = (model as any).contextWindow || CONTEXT_WINDOW_FALLBACK;
	if (!shouldCompact(tokens, contextWindow, SETTINGS)) return;

	const hasSummary = messages[0]?.role === "compactionSummary";
	const summaryStart = hasSummary ? 1 : 0;
	const cut = findCutIndex(messages, SETTINGS.keepRecentTokens);
	if (cut <= summaryStart) return; // 没有可总结的旧消息

	const toSummarize = messages.slice(summaryStart, cut);
	const recent = messages.slice(cut);
	const firstKept = getPgId(recent[0]);
	if (firstKept == null) {
		console.warn(`[compact] session ${sessionId}: first kept message has no pgId, skipping`);
		return;
	}
	const previousSummary = hasSummary ? (messages[0] as any).summary as string : undefined;

	const result = await generateSummaryWithUsage(
		toSummarize,
		models,
		model,
		SETTINGS.reserveTokens,
		undefined,
		previousSummary,
		undefined,
	);
	if (!result.ok) {
		console.warn(`[compact] session ${sessionId}: summarization failed: ${result.error}`);
		return;
	}

	await persistCompaction(pool, sessionId, result.value.text, firstKept, tokens);
	const newSummary = createCompactionSummaryMessage(result.value.text, tokens, new Date().toISOString());
	agent.state.messages = [newSummary, ...recent];
	console.log(
		`[compact] session ${sessionId}: ${tokens} tokens -> summary + ${recent.length} msgs (kept from msg#${firstKept})`,
	);
}
