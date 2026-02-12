import time
from datetime import datetime, timedelta, timezone

COIN_ORDER = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
COIN_SHORT = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP", "BNBUSDT": "BNB", "DOGEUSDT": "DOGE"}

TRADES_PER_PAGE = 10


def _fmt_pnl(val):
    sign = "+" if val >= 0 else ""
    icon = "\U0001f7e9" if val >= 0 else "\U0001f7e5"
    return f"{icon} {sign}{val:,.2f}$"


def _fmt_price(symbol, price_data):
    short = COIN_SHORT.get(symbol, symbol.replace("USDT", ""))
    last = price_data.get("last", 0)
    chg = price_data.get("change24h", 0)
    chg_icon = "\U0001f7e2" if chg >= 0 else "\U0001f534"
    chg_sign = "+" if chg >= 0 else ""
    if last >= 1000:
        p_str = f"${last:,.0f}"
    elif last >= 1:
        p_str = f"${last:,.2f}"
    else:
        p_str = f"${last:,.4f}"
    return f"{chg_icon} {short}: {p_str} ({chg_sign}{chg:.1f}%)"


def _duration_str(start_time, end_time=None):
    try:
        start_dt = datetime.fromisoformat(start_time) if isinstance(start_time, str) else start_time
        end_dt = datetime.fromisoformat(end_time) if isinstance(end_time, str) else (end_time or datetime.utcnow())
        delta = end_dt - start_dt
        hours = delta.seconds // 3600
        mins = (delta.seconds % 3600) // 60
        if delta.days > 0:
            return f"{delta.days}\u0434 {hours}\u0447 {mins}\u043c"
        elif hours > 0:
            return f"{hours}\u0447 {mins}\u043c"
        else:
            return f"{mins}\u043c"
    except (ValueError, TypeError):
        return "?"


def format_main_view(state):
    trading = state.trading_active
    positions = state.open_positions
    balance = state.demo_balance
    unrealized_pnl = sum(p.get("pnl", 0) for p in positions)

    lines = ["\U0001f916 AI Rvanka \u0422\u0440\u0435\u0439\u0434\u0438\u043d\u0433 \u0411\u043e\u0442"]
    lines.append("\u2501" * 28)
    lines.append(f"\U0001f4b0 \u0411\u0430\u043b\u0430\u043d\u0441: ${balance:,.2f}")

    if unrealized_pnl != 0:
        ur_sign = "+" if unrealized_pnl >= 0 else ""
        ur_icon = "\U0001f4c8" if unrealized_pnl >= 0 else "\U0001f4c9"
        total_equity = balance + unrealized_pnl
        lines.append(f"{ur_icon} \u041d\u0435\u0440\u0435\u0430\u043b. \u043f\u0440\u0438\u0431\u044b\u043b\u044c: {ur_sign}{unrealized_pnl:,.2f}$")
        lines.append(f"\U0001f48e \u041a\u0430\u043f\u0438\u0442\u0430\u043b: ${total_equity:,.2f}")

    lines.append("")
    if trading:
        session_num = state.session_counter
        lines.append(f"\U0001f7e2 \u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f \u0410\u041a\u0422\u0418\u0412\u041d\u0410 | \u0421\u0435\u0441\u0441\u0438\u044f #{session_num}")
        start_time = state.get("session_start_time")
        if start_time:
            lines.append(f"\u23f1 \u0414\u043b\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044c: {_duration_str(start_time)}")
    else:
        lines.append("\U0001f534 \u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f \u041e\u0421\u0422\u0410\u041d\u041e\u0412\u041b\u0415\u041d\u0410")

    max_pos = state.max_positions
    lines.append(f"\U0001f4ca \u041f\u043e\u0437\u0438\u0446\u0438\u0439: {len(positions)}/{max_pos}")

    if positions:
        lines.append("")
        for pos in positions:
            direction = pos.get("direction_str", "?")
            d_icon = "\U0001f535" if direction == "LONG" else "\U0001f534"
            d_ru = "\u041b\u041e\u041d\u0413" if direction == "LONG" else "\u0428\u041e\u0420\u0422"
            coin = pos.get("symbol", "???").replace("USDT", "")
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            lev = pos.get("leverage", 1)
            p_icon = "\U0001f7e9" if pnl >= 0 else "\U0001f7e5"
            sign = "+" if pnl >= 0 else ""
            sign_pct = "+" if pnl_pct >= 0 else ""
            trailing = " \U0001f4d0" if pos.get("trailing_activated") else ""
            tp = pos.get("tp_price", 0)
            sl = pos.get("sl_price", 0)
            lines.append(f"{d_icon} {d_ru} {coin} {lev}x{trailing}")
            lines.append(f"   ${entry:,.2f} \u2192 ${current:,.2f}")
            lines.append(f"   {p_icon} {sign}{pnl:,.2f}$ ({sign_pct}{pnl_pct:.2f}%)")
            lines.append(f"   \U0001f3af \u0422\u041f: ${tp:,.2f} | \U0001f6d1 \u0421\u041b: ${sl:,.2f}")
    elif trading:
        lines.append("")
        lines.append("\u23f3 \u041e\u0436\u0438\u0434\u0430\u043d\u0438\u0435 \u0441\u0438\u0433\u043d\u0430\u043b\u043e\u0432...")

    lines.append("")
    if trading:
        session_start_bal = state.get("session_start_balance", balance)
        session_pnl = balance - session_start_bal + unrealized_pnl
        session_pct = (session_pnl / max(session_start_bal, 1)) * 100
        s_icon = "\U0001f4c8" if session_pnl >= 0 else "\U0001f4c9"
        s_sign = "+" if session_pnl >= 0 else ""
        lines.append(f"{s_icon} \u0421\u0435\u0441\u0441\u0438\u044f: {s_sign}{session_pnl:,.2f}$ ({s_sign}{session_pct:.1f}%)")

    pnl_today = state.get_pnl_period(1) + unrealized_pnl
    pnl_7d = state.get_pnl_period(7) + unrealized_pnl

    lines.append("\u2501" * 28)
    lines.append(f"\U0001f4c5 \u0421\u0435\u0433\u043e\u0434\u043d\u044f: {_fmt_pnl(pnl_today)}")
    lines.append(f"\U0001f4c5 7 \u0434\u043d\u0435\u0439: {_fmt_pnl(pnl_7d)}")

    total_trades = state.get("stats", {}).get("total_trades", 0)
    wr = state.get_win_rate()
    lines.append(f"\U0001f3c6 \u0412\u0438\u043d\u0440\u0435\u0439\u0442: {wr:.1f}% | \u0412\u0441\u0435\u0433\u043e: {total_trades} \u0441\u0434.")
    lines.append("\u2501" * 28)

    live_prices = getattr(state, "_live_prices", None)
    if live_prices and trading:
        lines.append("")
        lines.append("\U0001f4b9 \u0420\u044b\u043d\u043e\u043a:")
        for sym in COIN_ORDER:
            if sym in live_prices:
                lines.append(_fmt_price(sym, live_prices[sym]))
        ts = getattr(state, "_last_update_ts", None)
        if ts:
            upd = datetime.fromtimestamp(ts, tz=timezone.utc)
            lines.append(f"\U0001f552 {upd.strftime('%H:%M:%S')} UTC")

    return "\n".join(lines)



def format_sessions_list(sessions):
    if not sessions:
        return (
            "\U0001f4cb \u0418\u0441\u0442\u043e\u0440\u0438\u044f \u0441\u0435\u0441\u0441\u0438\u0439\n\n"
            "\u0421\u0435\u0441\u0441\u0438\u0439 \u043f\u043e\u043a\u0430 \u043d\u0435 \u0431\u044b\u043b\u043e.\n"
            "\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0421\u0442\u0430\u0440\u0442 \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u0438."
        )

    total_pnl = sum(s.get("pnl", 0) for s in sessions)
    total_trades = sum(s.get("total_trades", 0) for s in sessions)
    total_wins = sum(s.get("wins", 0) for s in sessions)
    wr = round(total_wins / max(total_trades, 1) * 100, 1)

    lines = [
        "\U0001f4cb \u0418\u0441\u0442\u043e\u0440\u0438\u044f \u0441\u0435\u0441\u0441\u0438\u0439",
        "\u2501" * 28,
        f"\U0001f4ca \u0412\u0441\u0435\u0433\u043e \u0441\u0435\u0441\u0441\u0438\u0439: {len(sessions)}",
        f"\U0001f4b0 \u041e\u0431\u0449\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c: {_fmt_pnl(total_pnl)}",
        f"\U0001f3c6 \u0412\u0438\u043d\u0440\u0435\u0439\u0442: {wr:.1f}% | {total_trades} \u0441\u0434.",
        "\u2501" * 28,
        "",
        "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0435\u0441\u0441\u0438\u044e:",
    ]
    return "\n".join(lines)


def format_session_detail(session_data, trade_page=0):
    if not session_data:
        return "\u274c \u0421\u0435\u0441\u0441\u0438\u044f \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430", 0

    num = session_data.get("session_number", "?")
    start_time = session_data.get("start_time", "?")
    end_time = session_data.get("end_time", "?")
    start_bal = session_data.get("start_balance", 0)
    end_bal = session_data.get("end_balance", 0)
    pnl = session_data.get("pnl", 0)
    pnl_pct = session_data.get("pnl_pct", 0)
    total_trades = session_data.get("total_trades", 0)
    wins = session_data.get("wins", 0)
    wr = session_data.get("win_rate", 0)
    trades = session_data.get("trades", [])

    result_icon = "\U0001f7e9" if pnl >= 0 else "\U0001f7e5"
    dur = _duration_str(start_time, end_time)
    pnl_sign = "+" if pnl >= 0 else ""
    pct_sign = "+" if pnl_pct >= 0 else ""

    lines = [
        f"\U0001f4ca \u0421\u0435\u0441\u0441\u0438\u044f #{num}",
        "\u2501" * 28,
        f"\u23f1 \u0414\u043b\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044c: {dur}",
        f"\U0001f4b0 ${start_bal:,.2f} \u2192 ${end_bal:,.2f}",
        f"{result_icon} \u041f\u0440\u0438\u0431\u044b\u043b\u044c: {pnl_sign}{pnl:,.2f}$ ({pct_sign}{pnl_pct:.1f}%)",
        f"\U0001f3c6 \u0412\u0438\u043d\u0440\u0435\u0439\u0442: {wr:.1f}% | {total_trades} \u0441\u0434. ({wins}\u0412/{total_trades - wins}\u041f)",
    ]

    total_trade_pages = max(1, (len(trades) + TRADES_PER_PAGE - 1) // TRADES_PER_PAGE)

    tp_count = sum(1 for t in trades if t.get("reason") == "TP")
    sl_count = sum(1 for t in trades if t.get("reason") == "SL")
    trail_count = sum(1 for t in trades if t.get("reason") == "TRAILING_SL")
    if total_trades > 0:
        lines.append(f"\U0001f3af \u0422\u041f: {tp_count} | \U0001f6d1 \u0421\u041b: {sl_count} | \U0001f4d0 \u0422\u0440\u0435\u0439\u043b: {trail_count}")

    coin_summary = {}
    for t in trades:
        sym = t.get("symbol", "?")
        if sym not in coin_summary:
            coin_summary[sym] = {"trades": 0, "pnl": 0, "wins": 0}
        coin_summary[sym]["trades"] += 1
        coin_summary[sym]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            coin_summary[sym]["wins"] += 1

    if coin_summary:
        lines.append("")
        lines.append("\U0001f4ca \u041f\u043e \u043c\u043e\u043d\u0435\u0442\u0430\u043c:")
        for sym, data in sorted(coin_summary.items(), key=lambda x: x[1]["pnl"], reverse=True):
            coin_name = sym.replace("USDT", "")
            wr_coin = round(data["wins"] / max(data["trades"], 1) * 100, 1)
            p_icon = "\U0001f7e9" if data["pnl"] >= 0 else "\U0001f7e5"
            c_sign = "+" if data["pnl"] >= 0 else ""
            coin_pnl = data["pnl"]
            coin_tr = data["trades"]
            lines.append(
                f"  {coin_name}: {p_icon} {c_sign}{coin_pnl:,.2f}$ | {coin_tr} \u0441\u0434. | {wr_coin:.0f}%"
            )

    if trades:
        start_idx = trade_page * TRADES_PER_PAGE
        end_idx = min(start_idx + TRADES_PER_PAGE, len(trades))
        page_trades = trades[start_idx:end_idx]

        lines.append("")
        if total_trade_pages > 1:
            lines.append(f"\U0001f4ca \u0421\u0434\u0435\u043b\u043a\u0438 ({start_idx + 1}-{end_idx} \u0438\u0437 {len(trades)}):")
        else:
            lines.append(f"\U0001f4ca \u0421\u0434\u0435\u043b\u043a\u0438 ({len(trades)}):")

        for t in page_trades:
            sym = t.get("symbol", "?").replace("USDT", "")
            d = t.get("direction_str", "?")
            d_icon = "\U0001f535" if d == "LONG" else "\U0001f534"
            d_ru = "\u041b\u041e\u041d\u0413" if d == "LONG" else "\u0428\u041e\u0420\u0422"
            t_pnl = t.get("pnl", 0)
            t_icon = "\U0001f7e9" if t_pnl >= 0 else "\U0001f7e5"
            reason = t.get("reason", "?")
            reason_map = {
                "TP": "\U0001f3af\u0422\u041f", "SL": "\U0001f6d1\u0421\u041b",
                "TRAILING_SL": "\U0001f4d0\u0422\u0440\u0435\u0439\u043b", "SESSION_END": "\u23f9\u0421\u0442\u043e\u043f",
            }
            r_icon = reason_map.get(reason, "\u2753")
            entry_p = t.get("entry_price", 0)
            exit_p = t.get("exit_price", 0)
            lev = t.get("leverage", 1)
            trail = " \U0001f4d0" if t.get("trailing_activated") else ""
            t_sign = "+" if t_pnl >= 0 else ""
            lines.append(f"  {d_icon} {d_ru} {sym} {lev}x | {r_icon}{trail}")
            lines.append(f"     ${entry_p:,.2f} \u2192 ${exit_p:,.2f} | {t_icon} {t_sign}{t_pnl:,.2f}$")

    return "\n".join(lines), total_trade_pages
