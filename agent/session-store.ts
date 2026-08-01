/**
 * agent_session 的 PG 读写：加载历史(冷启动还原记忆) + 持久化全量 message_json + compaction 查询。
 *
 * message_json 存全量 AgentMessage（含 toolCall 块/usage），冷启动按它精确还原 _state.messages。
 * 旧列(role/content/tool_*)保留给 webui 展示，从 message 派生。
 * PG id 用 Symbol 挂在 message 上（非枚举，不进 message_json），供 compact 的 first_kept_message_id 用。
 */
import type { AgentMessage } from "./src/types.ts";
import { createCompactionSummaryMessage } from "./src/harness/messages.ts";
import { contentText } from "@earendil-works/pi-ai";

const PG_ID = Symbol("__pgId");

export function getPgId(msg: AgentMessage): number | undefined {
	return (msg as any)[PG_ID];
}

function setPgId(msg: AgentMessage, id: number): void {
	Object.defineProperty(msg, PG_ID, {
		value: id,
		enumerable: false,
		writable: true,
		configurable: true,
	});
}

/** 冷启动：加载 session 的历史 messages（message_json）+ 最新 compaction 摘要。
 * 只加载 first_kept_message_id 之后的（之前的已被总结进 summary）。 */
export async function loadSession(pool: any, sessionId: number): Promise<{
	messages: AgentMessage[];
	compactionSummary: AgentMessage | null;
}> {
	const comp = await pool.query(
		"SELECT summary, first_kept_message_id, tokens_before, created_at FROM agent_session.compaction WHERE session_id=$1 ORDER BY created_at DESC LIMIT 1",
		[sessionId],
	);
	const firstKept: number = comp.rows[0]?.first_kept_message_id ?? 0;
	const res = await pool.query(
		"SELECT id, message_json FROM agent_session.message WHERE session_id=$1 AND message_json IS NOT NULL AND id >= $2 ORDER BY created_at",
		[sessionId, firstKept],
	);
	const messages: AgentMessage[] = [];
	for (const row of res.rows) {
		try {
			// pg 库读 jsonb 列默认已 JSON.parse 成对象，不能再 parse 一次
			const msg = typeof row.message_json === "string"
				? (JSON.parse(row.message_json) as AgentMessage)
				: (row.message_json as AgentMessage);
			setPgId(msg, row.id);
			messages.push(msg);
		} catch {
			// 跳过无法解析的行
		}
	}
	let compactionSummary: AgentMessage | null = null;
	if (comp.rows[0]) {
		compactionSummary = createCompactionSummaryMessage(
			comp.rows[0].summary,
			comp.rows[0].tokens_before ?? 0,
			comp.rows[0].created_at instanceof Date
				? comp.rows[0].created_at.toISOString()
				: comp.rows[0].created_at,
		);
	}
	return { messages, compactionSummary };
}

/** 持久化一条 message（全量 message_json + 派生展示列）。返回 PG id 并挂在 msg 上。
 * toolArgs 仅 toolResult 用（从 tool_execution_start 事件捕获，toolResult message 本身不带 args）。 */
export async function persistMessage(
	pool: any,
	sessionId: number,
	msg: AgentMessage,
	toolArgs?: unknown,
): Promise<number | null> {
	let role: string = msg.role;
	let content: string | null = null;
	let displayContent: string | null = null;
	let toolName: string | null = null;
	let toolArgsJson: string | null = null;
	let toolResultJson: string | null = null;

	if (msg.role === "user" || msg.role === "assistant") {
		content = contentText((msg as any).content);
		displayContent = content;
	} else if (msg.role === "toolResult") {
		role = "tool_call";
		toolName = (msg as any).toolName ?? null;
		toolArgsJson = toolArgs != null ? JSON.stringify(toolArgs) : null;
		const details = (msg as any).details;
		toolResultJson = JSON.stringify(details ?? contentText((msg as any).content));
	} else {
		// compactionSummary / branchSummary 等不持久化为 message（compactionSummary 由 compaction 表管）
		return null;
	}

	const r = await pool.query(
		"INSERT INTO agent_session.message(session_id, role, content, display_content, tool_name, tool_args, tool_result, message_json) "
		+ "VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb) RETURNING id",
		[sessionId, role, content, displayContent, toolName, toolArgsJson, toolResultJson, JSON.stringify(msg)],
	);
	const id = r.rows[0]?.id;
	if (id != null) setPgId(msg, id);
	return id ?? null;
}

/** 持久化一次 compact 摘要（append-only；first_kept_message_id 之后的 message 保留原文）。 */
export async function persistCompaction(
	pool: any,
	sessionId: number,
	summary: string,
	firstKeptMessageId: number,
	tokensBefore: number,
): Promise<void> {
	await pool.query(
		"INSERT INTO agent_session.compaction(session_id, summary, first_kept_message_id, tokens_before) VALUES($1,$2,$3,$4)",
		[sessionId, summary, firstKeptMessageId, tokensBefore],
	);
}
