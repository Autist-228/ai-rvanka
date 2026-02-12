from datetime import datetime, timedelta


def _fmt_pnl(val):
    sign = "+" if val >= 0 else ""
    icon = "\U0001f7e9" if val >= 0 else "\U0001f7e5"
    return f"{icon} {sign}{val:,.2f}$"


def format_main_view(state):
    is_demo = state.active_account == "demo"
    trading = state.trading_active
    positions = state.open_positions
    balance = state.demo_balance

    unrealized_pnl = sum(p.get("pnl", 0) for p in positions)

    acc_icon = "\U0001f7e2" if is_demo else "\U0001f535"
    acc_name = "\u0414\u0415\u041c\u041e" if is_demo else "\u0420\u0415\u0410\u041b"

    lines = [f"{acc_icon} {acc_name} | \U0001f4b0 ${balance:,.2f}"]

    if unrealized_pnl != 0:
        ur_sign = "+" if unrealized_pnl >= 0 else ""
        ur_icon = "\U0001f4c8" if unrealized_pnl >= 0 else "\U0001f4c9"
        total_equity = balance + unrealized_pnl
        lines.append(f"{ur_icon} \u041d\u0435\u0440\u0435\u0430\u043b. P&L: {ur_sign}{unrealized_pnl:,.2f}$")
        lines.append(f"\U0001f48e Equity: ${total_equity:,.2f}")

    lines.append("")
    if trading:
        lines.append("\U0001f7e2 \u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f \u0410\u041a\u0422\u0418\u0412\u041d\u0410")
    else:
        lines.append("\U0001f534 \u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f \u041e\u0421\u0422\u0410\u041d\u041e\u0412\u041b\u0415\u041d\u0410")

    max_pos = state.settings.get("max_concurrent_total", 3)
    lines.append(f"\U0001f4ca \u0421\u0434\u0435\u043b\u043e\u043a: {len(positions)}/{max_pos}")

    if positions:
        lines.append("")
        for pos in positions:
            direction = pos.get("direction_str", "?")
            d_icon = "\U0001f535" if direction == "LONG" else "\U0001f534"
            coin = pos.get("symbol", "???").replace("USDT", "")
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            lev = pos.get("leverage", 1)
            size = pos.get("size_usd", 0)
            p_icon = "\U0001f7e9" if pnl >= 0 else "\U0001f7e5"
            sign = "+" if pnl >= 0 else ""
            sign_pct = "+" if pnl_pct >= 0 else ""
            lines.append(f"{d_icon} {direction} {coin} {lev}x | ${entry:,.2f} \u2192 ${current:,.2f}")
            lines.append(f"   {p_icon} {sign}{pnl:,.2f}$ ({sign_pct}{pnl_pct:.2f}%) | ${size:.0f}")
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
        lines.append(f"{s_icon} \u0422\u0435\u043a\u0443\u0449\u0430\u044f \u0441\u0435\u0441\u0441\u0438\u044f: {s_sign}{session_pnl:,.2f}$ ({s_sign}{session_pct:.1f}%)")
    else:
        last = state.get("last_session")
        if last:
            pnl = last.get("pnl", 0)
            pct = last.get("pnl_pct", 0)
            p_icon = "\U0001f4c8" if pnl >= 0 else "\U0001f4c9"
            p_sign = "+" if pnl >= 0 else ""
            lines.append(f"{p_icon} \u041f\u0440\u0435\u0434. \u0441\u0435\u0441\u0441\u0438\u044f: {p_sign}{pnl:,.2f}$ ({p_sign}{pct:.1f}%)")
        else:
            lines.append("\U0001f4ca \u0421\u0435\u0441\u0441\u0438\u0439 \u043f\u043e\u043a\u0430 \u043d\u0435 \u0431\u044b\u043b\u043e")

    pnl_today = state.get_pnl_period(1) + unrealized_pnl
    pnl_7d = state.get_pnl_period(7) + unrealized_pnl
    pnl_30d = state.get_pnl_period(30) + unrealized_pnl

    lines.append("")
    lines.append("\u2501" * 24)
    lines.append(f"\U0001f4c5 \u0421\u0435\u0433\u043e\u0434\u043d\u044f:  {_fmt_pnl(pnl_today)}")
    lines.append(f"\U0001f4c5 7 \u0434\u043d\u0435\u0439:    {_fmt_pnl(pnl_7d)}")
    lines.append(f"\U0001f4c5 30 \u0434\u043d\u0435\u0439:  {_fmt_pnl(pnl_30d)}")
    lines.append("\u2501" * 24)
    lines.append("")

    total_trades = state.get("stats", {}).get("total_trades", 0)
    wr = state.get_win_rate()
    lines.append(f"\U0001f3c6 WR: {wr:.1f}% | \u0421\u0434\u0435\u043b\u043e\u043a: {total_trades}")

    return "\n".join(lines)


def format_open_positions(state):
    positions = state.open_positions

    if not positions:
        return "📭 Нет открытых сделок\n\nБот ожидает сигналы от модели..."

    lines = [f"📊 Открытые сделки ({len(positions)}):\n"]

    for pos in positions:
        direction = pos.get("direction_str", "?")
        icon = "🔵" if direction == "LONG" else "🔴"
        symbol = pos.get("symbol", "???")
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", entry)
        size = pos.get("size_usd", 0)
        leverage = pos.get("leverage", 1)
        pnl = pos.get("pnl", 0)
        pnl_pct = pos.get("pnl_pct", 0)
        tp = pos.get("tp_price", 0)
        sl = pos.get("sl_price", 0)
        tp_pct_val = pos.get("tp_pct", 0) * 100
        sl_pct_val = pos.get("sl_pct", 0) * 100
        conf = pos.get("confidence", 0)

        open_time = pos.get("open_time", "")
        duration = ""
        if open_time:
            try:
                opened = datetime.fromisoformat(open_time)
                delta = datetime.utcnow() - opened
                hours = delta.seconds // 3600
                mins = (delta.seconds % 3600) // 60
                if delta.days > 0:
                    duration = f"{delta.days}д {hours}ч"
                elif hours > 0:
                    duration = f"{hours}ч {mins}м"
                else:
                    duration = f"{mins}м"
            except (ValueError, TypeError):
                duration = "?"

        pnl_icon = "🟩" if pnl >= 0 else "🟥"

        coin_name = symbol.replace("USDT", "")

        lines.append(
            f"{icon} {direction} {coin_name}/USDT\n"
            f"├ Вход: ${entry:,.4f} → Сейчас: ${current:,.4f}\n"
            f"├ Размер: ${size:.2f} ({leverage}x) | Уверенность: {conf:.1%}\n"
            f"├ {pnl_icon} P&L: {'+' if pnl >= 0 else ''}{pnl:,.2f}$ ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)\n"
            f"├ 🎯 TP: ${tp:,.4f} (+{tp_pct_val:.1f}%)\n"
            f"├ 🛑 SL: ${sl:,.4f} (-{sl_pct_val:.1f}%)\n"
            f"└ ⏱ Открыта: {duration} назад\n"
        )

    return "\n".join(lines)


def format_last_session(state):
    last = state.get("last_session")

    if not last:
        return "\U0001f4cb \u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u043e \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0435\u0439 \u0441\u0435\u0441\u0441\u0438\u0438\n\n\u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044e, \u0447\u0442\u043e\u0431\u044b \u0441\u043e\u0437\u0434\u0430\u0442\u044c \u0441\u0435\u0441\u0441\u0438\u044e."

    start_time = last.get("start_time", "?")
    end_time = last.get("end_time", "?")
    start_bal = last.get("start_balance", 0)
    end_bal = last.get("end_balance", 0)
    pnl = last.get("pnl", 0)
    pnl_pct = last.get("pnl_pct", 0)
    trades_count = last.get("trades", 0)

    result_icon = "\U0001f7e9" if pnl >= 0 else "\U0001f7e5"

    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = end_dt - start_dt
        hours = duration.seconds // 3600
        mins = (duration.seconds % 3600) // 60
        if duration.days > 0:
            dur_str = f"{duration.days}\u0434 {hours}\u0447 {mins}\u043c"
        else:
            dur_str = f"{hours}\u0447 {mins}\u043c"
    except (ValueError, TypeError):
        dur_str = "?"

    session_trades = [t for t in state.trade_history if t.get("session_start") == start_time]

    lines = [
        "\U0001f4cb \u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u0441\u0435\u0441\u0441\u0438\u044f",
        "\u2501" * 28,
        f"\u23f1 \u0414\u043b\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044c: {dur_str}",
        f"\U0001f4b0 \u0411\u0430\u043b\u0430\u043d\u0441: ${start_bal:,.2f} \u2192 ${end_bal:,.2f}",
        f"{result_icon} \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442: {'+' if pnl >= 0 else ''}{pnl:,.2f}$ ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)",
        f"\U0001f4ca \u0421\u0434\u0435\u043b\u043e\u043a: {trades_count}",
        "",
        "\u2501" * 28,
        "\U0001f4ca \u0421\u0434\u0435\u043b\u043a\u0438 \u0441\u0435\u0441\u0441\u0438\u0438:",
    ]

    if session_trades:
        for t in session_trades:
            sym = t.get("symbol", "?").replace("USDT", "")
            d = t.get("direction_str", "?")
            d_icon = "\U0001f535" if d == "LONG" else "\U0001f534"
            t_pnl = t.get("pnl", 0)
            t_icon = "\U0001f7e9" if t_pnl >= 0 else "\U0001f7e5"
            reason = t.get("reason", "?")
            reason_map = {"TP": "\U0001f3af", "SL": "\U0001f6d1", "TRAILING_SL": "\U0001f4d0", "SESSION_END": "\u23f9"}
            r_icon = reason_map.get(reason, "\u2753")
            entry_p = t.get("entry_price", 0)
            exit_p = t.get("exit_price", 0)
            lev = t.get("leverage", 1)
            lines.append(f"  {d_icon} {d} {sym} {lev}x | {r_icon} {reason}")
            lines.append(f"     ${entry_p:,.2f} \u2192 ${exit_p:,.2f} | {t_icon} {'+' if t_pnl >= 0 else ''}{t_pnl:,.2f}$")
    else:
        lines.append("  \u041d\u0435\u0442 \u0441\u0434\u0435\u043b\u043e\u043a")

    lines.append("")

    coin_summary = {}
    for t in session_trades:
        sym = t.get("symbol", "?")
        if sym not in coin_summary:
            coin_summary[sym] = {"trades": 0, "pnl": 0, "wins": 0}
        coin_summary[sym]["trades"] += 1
        coin_summary[sym]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            coin_summary[sym]["wins"] += 1

    if coin_summary:
        lines.append("\u2501" * 28)
        lines.append("\U0001f4ca \u0418\u0442\u043e\u0433\u043e \u043f\u043e \u043c\u043e\u043d\u0435\u0442\u0430\u043c:")
        for sym, data in sorted(coin_summary.items(), key=lambda x: x[1]["pnl"], reverse=True):
            coin_name = sym.replace("USDT", "")
            wr = round(data["wins"] / max(data["trades"], 1) * 100, 1)
            p_icon = "\U0001f7e9" if data["pnl"] >= 0 else "\U0001f7e5"
            lines.append(
                f"  {p_icon} {coin_name}: {data['trades']} \u0441\u0434. | "
                f"WR {wr:.0f}% | {'+' if data['pnl'] >= 0 else ''}{data['pnl']:,.2f}$"
            )

    return "\n".join(lines)


def format_settings(state):
    s = state.settings

    lines = [
        f"⚙️ Настройки\n",
        f"{'━' * 28}\n\n",
        f"💰 Демо-баланс: ${state.demo_balance:,.2f}\n",
        f"🎯 Уверенность: {s.get('confidence_threshold', 0.65):.0%}\n",
        f"⚡ Макс. leverage: {s.get('max_leverage', 20)}x\n",
        f"📏 Макс. позиция: {s.get('max_position_pct', 0.05):.0%} / ${s.get('max_position_usd', 300):.0f}\n",
        f"📊 Макс. сделок: {s.get('max_concurrent_total', 3)}\n",
        f"📉 Стоп при просадке: {s.get('max_drawdown_pct', 0.15):.0%}\n",
        f"⏸ Пауза после лузов: {s.get('consecutive_sl_pause', 3)}\n\n",
        f"{'━' * 28}\n",
        f"🎯 TP/SL по монетам:\n",
    ]

    coin_tpsl = s.get("coin_tp_sl", {})
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]:
        coin = symbol.replace("USDT", "")
        tp_sl = coin_tpsl.get(symbol, {})
        tp = tp_sl.get("tp_pct", 0.04) * 100
        sl = tp_sl.get("sl_pct", 0.02) * 100
        lines.append(f"  {coin}: TP {tp:.1f}% / SL {sl:.1f}%\n")

    return "".join(lines)


def format_setting_edit(setting_key, current_value):
    labels = {
        "confidence_threshold": ("🎯 Уверенность сигнала", "Мин. confidence для входа в сделку"),
        "max_leverage": ("⚡ Макс. leverage", "Максимальное плечо"),
        "max_position_pct": ("📏 Макс. позиция %", "Макс. % от баланса на сделку"),
        "max_position_usd": ("📏 Макс. позиция $", "Макс. $ на сделку"),
        "max_concurrent_total": ("📊 Макс. одновременных сделок", "Кол-во одновременных позиций"),
        "max_drawdown_pct": ("📉 Стоп при просадке", "Макс. просадка портфеля для стопа"),
        "consecutive_sl_pause": ("⏸ Пауза после лузов", "Кол-во SL подряд для паузы на монете"),
        "demo_balance": ("💰 Демо-баланс", "Баланс демо-счёта"),
    }

    label, desc = labels.get(setting_key, (setting_key, ""))

    if isinstance(current_value, float) and current_value < 1:
        val_str = f"{current_value:.0%}"
    else:
        val_str = str(current_value)

    return (
        f"{label}\n"
        f"{'━' * 28}\n\n"
        f"📝 {desc}\n"
        f"Текущее значение: {val_str}\n\n"
        f"Выберите новое значение:"
    )


def format_coin_tpsl_edit(symbol, current_tp, current_sl):
    coin = symbol.replace("USDT", "")
    return (
        f"🎯 TP/SL для {coin}\n"
        f"{'━' * 28}\n\n"
        f"Take Profit: {current_tp * 100:.1f}%\n"
        f"Stop Loss: {current_sl * 100:.1f}%\n\n"
        f"Выберите что изменить:"
    )
