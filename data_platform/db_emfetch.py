"""em-fetch 数据入库（market_data + event_data，market_data_writer 账号）。

10 种数据类型：stock_meta / stock_daily / stock_adj_factor / fund_flow_daily /
industry_daily / dragon_tiger / dragon_tiger_seats / lockup_expiry / stock_news / macro_news

ON CONFLICT DO NOTHING（重抓不重复），stock_meta 用 UPSERT（ON CONFLICT DO UPDATE）。
"""
import asyncio
import asyncpg
import os
from datetime import date as _date, datetime as _datetime

_md_pool: asyncpg.Pool | None = None
_md_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_market_data_pool() -> asyncpg.Pool:
    """market_data_writer 连接池（INSERT/SELECT/UPDATE market_data + event_data）。"""
    global _md_pool, _md_pool_loop
    loop = asyncio.get_running_loop()
    if _md_pool is None or _md_pool_loop is not loop:
        if _md_pool is not None:
            try:
                await _md_pool.close()
            except Exception:
                pass
        _md_pool = await asyncpg.create_pool(
            host=os.environ.get("DP_PG_HOST", "localhost"),
            port=int(os.environ.get("DP_PG_PORT", "5432")),
            user="market_data_writer",
            password=os.environ.get("MARKET_DATA_WRITER_PWD", "mdw_2026"),
            database=os.environ.get("DP_PG_DB", "rag"),
            min_size=2,
            max_size=5,
        )
        _md_pool_loop = loop
    return _md_pool


def _derive_board(code: str) -> str:
    """按 code 前缀派生板块。"""
    if code.startswith("688"):
        return "star"
    if code.startswith("300"):
        return "gem"
    if code.startswith("6"):
        return "sh_main"
    if code.startswith("00"):
        return "sz_main"
    if code.startswith("8") or code.startswith("4"):
        return "bjse"
    return "sh_main"


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


async def _ingest_stock_meta(conn: asyncpg.Connection, meta: dict) -> int:
    """stock_meta: UPSERT（code 冲突 -> UPDATE + updated_at）。"""
    code = meta.get("code")
    if not code:
        return 0
    await conn.execute(
        """
        INSERT INTO market_data.stock_meta
            (code, name, industry, total_shares, float_shares, list_date, board, is_active, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, true, now())
        ON CONFLICT (code) DO UPDATE SET
            name=EXCLUDED.name, industry=EXCLUDED.industry,
            total_shares=EXCLUDED.total_shares, float_shares=EXCLUDED.float_shares,
            list_date=EXCLUDED.list_date, board=EXCLUDED.board,
            is_active=true, updated_at=now()
        """,
        code,
        meta.get("name"),
        meta.get("industry"),
        _to_int(meta.get("total_shares")),
        _to_int(meta.get("float_shares")),
        meta.get("list_date"),
        _derive_board(code),
    )
    return 1


async def _ingest_stock_daily(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO market_data.stock_daily (code, trade_date, open, high, low, close, volume, amount)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        ON CONFLICT (code, trade_date) DO NOTHING
        """,
        [
            (r["code"], r["trade_date"], r.get("open"), r.get("high"),
             r.get("low"), r.get("close"), r.get("volume"), r.get("amount"))
            for r in rows
        ],
    )
    return len(rows)


async def _ingest_stock_adj_factor(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO market_data.stock_adj_factor (code, trade_date, adj_factor)
        VALUES ($1,$2,$3)
        ON CONFLICT (code, trade_date) DO NOTHING
        """,
        [(r["code"], r["trade_date"], r["adj_factor"]) for r in rows],
    )
    return len(rows)


async def _ingest_fund_flow_daily(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO market_data.fund_flow_daily (code, trade_date, main_net, super_net, large_net, mid_net, small_net)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (code, trade_date) DO NOTHING
        """,
        [
            (r["code"], r["trade_date"], r.get("main_net"), r.get("super_net"),
             r.get("large_net"), r.get("mid_net"), r.get("small_net"))
            for r in rows
        ],
    )
    return len(rows)


async def _ingest_industry_daily(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO market_data.industry_daily
            (trade_date, industry_code, industry_name, pct_change, up_count, down_count, flat_count, leader_code, leader_name)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (trade_date, industry_code) DO NOTHING
        """,
        [
            (r["trade_date"], r["industry_code"], r["industry_name"], r.get("pct_change"),
             r.get("up_count"), r.get("down_count"), None,
             r.get("leader_code"), r.get("leader_name"))
            for r in rows
        ],
    )
    return len(rows)


async def _ingest_dragon_tiger(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO event_data.dragon_tiger (code, trade_date, reason, net_buy, turnover_rate)
        VALUES ($1,$2,$3,$4,$5)
        ON CONFLICT (code, trade_date) DO NOTHING
        """,
        [
            (r["code"], r["trade_date"], r.get("reason"),
             r.get("net_buy"), r.get("turnover_rate"))
            for r in rows
        ],
    )
    return len(rows)


async def _ingest_dragon_tiger_seats(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    count = 0
    for r in rows:
        # 先查 dragon_tiger_id（按 code+trade_date）
        dt_id = await conn.fetchval(
            "SELECT id FROM event_data.dragon_tiger WHERE code=$1 AND trade_date=$2",
            r["code"], r["trade_date"],
        )
        if not dt_id:
            continue  # dragon_tiger 不存在，跳过
        await conn.execute(
            """
            INSERT INTO event_data.dragon_tiger_seat
                (dragon_tiger_id, code, trade_date, side, seat_name, is_institution, buy_amt, sell_amt, net_amt)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (code, trade_date, side, seat_name) DO NOTHING
            """,
            dt_id, r["code"], r["trade_date"], r["side"], r.get("seat_name"),
            r.get("is_institution", False), r.get("buy_amt"),
            r.get("sell_amt"), r.get("net_amt"),
        )
        count += 1
    return count


async def _ingest_lockup_expiry(conn: asyncpg.Connection, rows: list) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO event_data.lockup_expiry
            (code, free_date, stock_type, free_shares, free_ratio, is_upcoming, snapshot_date)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (code, free_date, stock_type, snapshot_date) DO NOTHING
        """,
        [
            (r["code"], r["free_date"], r.get("stock_type"),
             _to_int(r.get("free_shares")), r.get("free_ratio"),
             r.get("is_upcoming"), r.get("snapshot_date"))
            for r in rows
        ],
    )
    return len(rows)


async def _ingest_news(conn: asyncpg.Connection, rows: list, news_type: str) -> int:
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO event_data.stock_news (code, news_type, title, content, published_at, source, url)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (url) DO NOTHING
        """,
        [
            (r.get("code"), news_type, r.get("title"), r.get("content"),
             r.get("published_at"), r.get("source"), r.get("url"))
            for r in rows
        ],
    )
    return len(rows)


_DATE_FIELDS = {
    "stock_meta": ["list_date"],
    "stock_daily": ["trade_date"],
    "stock_adj_factor": ["trade_date"],
    "fund_flow_daily": ["trade_date"],
    "industry_daily": ["trade_date"],
    "dragon_tiger": ["trade_date"],
    "dragon_tiger_seats": ["trade_date"],
    "lockup_expiry": ["free_date", "snapshot_date"],
}
_DATETIME_FIELDS = {
    "stock_news": ["published_at"],
    "macro_news": ["published_at"],
}


def _preprocess_dates(data: dict) -> None:
    """把 em-fetch JSON 里的日期字符串转成 date/datetime 对象（asyncpg 要求）。"""
    for key, fields in _DATE_FIELDS.items():
        items = data.get(key)
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                continue
            for f in fields:
                v = item.get(f)
                if isinstance(v, str) and v:
                    try:
                        item[f] = _date.fromisoformat(v)
                    except ValueError:
                        item[f] = None
    for key, fields in _DATETIME_FIELDS.items():
        items = data.get(key)
        if not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for f in fields:
                v = item.get(f)
                if isinstance(v, str) and v:
                    try:
                        item[f] = _datetime.fromisoformat(v)
                    except ValueError:
                        item[f] = None


async def em_fetch_ingest_pg(data: dict) -> dict:
    """批量入库 em-fetch JSON（10 种数据类型）。返回 {type: count}。"""
    _preprocess_dates(data)
    pool = await get_market_data_pool()
    results = {}
    async with pool.acquire() as conn:
        if data.get("stock_meta"):
            results["stock_meta"] = await _ingest_stock_meta(conn, data["stock_meta"])
        if data.get("stock_daily"):
            results["stock_daily"] = await _ingest_stock_daily(conn, data["stock_daily"])
        if data.get("stock_adj_factor"):
            results["stock_adj_factor"] = await _ingest_stock_adj_factor(conn, data["stock_adj_factor"])
        if data.get("fund_flow_daily"):
            results["fund_flow_daily"] = await _ingest_fund_flow_daily(conn, data["fund_flow_daily"])
        if data.get("industry_daily"):
            results["industry_daily"] = await _ingest_industry_daily(conn, data["industry_daily"])
        if data.get("dragon_tiger"):
            results["dragon_tiger"] = await _ingest_dragon_tiger(conn, data["dragon_tiger"])
        if data.get("dragon_tiger_seats"):
            results["dragon_tiger_seats"] = await _ingest_dragon_tiger_seats(conn, data["dragon_tiger_seats"])
        if data.get("lockup_expiry"):
            results["lockup_expiry"] = await _ingest_lockup_expiry(conn, data["lockup_expiry"])
        if data.get("stock_news"):
            results["stock_news"] = await _ingest_news(conn, data["stock_news"], "stock_news")
        if data.get("macro_news"):
            results["macro_news"] = await _ingest_news(conn, data["macro_news"], "macro_wire")
    return results
