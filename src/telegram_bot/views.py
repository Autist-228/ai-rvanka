from datetime import datetime, timedelta


def format_main_view(state):
    is_demo = state.active_account == "demo"
    trading = state.trading_active
    positions = state.open_positions

    if is_demo:
        balance = state.demo_balance
        acc_line = "🟢 ДЕМО-СЧЁТ (активен)"
        acc_balance = f"💰 ${balance:,.2f}"
        bybit_line = "⚪ Реальный Bybit"
        bybit_balance = "💰 —"
    else:
        balance = 0.0
        acc_line = "⚪ Демо-счёт"
        acc_balance = f"💰 ${state.demo_balance:,.2f}"
        bybit_line = "🟢 Реальный Bybit (активен)"
        bybit_balance = "💰 подключите API"

    if trading:
        status_icon = "🟢"
        status_text = "Торговля АКТИВНА"
    else:
        status_icon = "🔴"
        status_text = "Торговля ОСТАНОВЛЕНА"

    session_line = ""
    if trading:
        session_start_bal = state.get("session_start_balance", balance)
        session_pnl = balance - session_start_bal
        session_pct = (session_pnl / max(session_start_bal, 1)) * 100
        pnl_icon = "📈" if session_pnl >= 0 else "📉"
        session_line = f"{pnl_icon} Текущая сессия: {'+' if session_pnl >= 0 else ''}{session_pnl:,.2f}$ ({'+' if session_pct >= 0 else ''}{session_pct:.1f}%)"
    else:
        last = state.get("last_session")
        if last:
            pnl = last.get("pnl", 0)
            pct = last.get("pnl_pct", 0)
            pnl_icon = "📈" if pnl >= 0 else "📉"
            session_line = f"{pnl_icon} Последняя сессия: {'+' if pnl >= 0 else ''}{pnl:,.2f}$ ({'+' if pct >= 0 else ''}{pct:.1f}%)"
        else:
            session_line = "📊 Сессий пока не было"

    pnl_today = state.get_pnl_period(1)
    pnl_7d = state.get_pnl_period(7)
    pnl_30d = state.get_pnl_period(30)

    def fmt_pnl(val):
        icon = "🟩" if val >= 0 else "🟥"
        return f"{icon} {'+' if val >= 0 else ''}{val:,.2f}$"

    text = (
        f"{'━' * 28}\n"
        f"  {acc_line}\n"
        f"  {acc_balance}\n"
        f"  {bybit_line}\n"
        f"  {bybit_balance}\n"
        f"{'━' * 28}\n\n"
        f"{status_icon} Статус: {status_text}\n"
        f"📊 Открыто сделок: {len(positions)}/{state.settings.get('max_concurrent_total', 3)}\n\n"
        f"{session_line}\n\n"
        f"📅 Сегодня: {fmt_pnl(pnl_today)}\n"
        f"📅 7 дней: {fmt_pnl(pnl_7d)}\n"
        f"📅 30 дней: {fmt_pnl(pnl_30d)}\n\n"
        f"🏆 Win Rate: {state.get_win_rate():.1f}% | "
        f"Сделок всего: {state.get('stats', {}).get('total_trades', 0)}"
    )

    return text


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
        return "📋 Нет данных о последней сессии\n\nЗапустите торговлю, чтобы создать сессию."

    start_time = last.get("start_time", "?")
    end_time = last.get("end_time", "?")
    start_bal = last.get("start_balance", 0)
    end_bal = last.get("end_balance", 0)
    pnl = last.get("pnl", 0)
    pnl_pct = last.get("pnl_pct", 0)
    trades_count = last.get("trades", 0)

    pnl_icon = "📈" if pnl >= 0 else "📉"
    result_icon = "🟩" if pnl >= 0 else "🟥"

    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = end_dt - start_dt
        hours = duration.seconds // 3600
        mins = (duration.seconds % 3600) // 60
        if duration.days > 0:
            dur_str = f"{duration.days}д {hours}ч {mins}м"
        else:
            dur_str = f"{hours}ч {mins}м"
    except (ValueError, TypeError):
        dur_str = "?"

    session_trades = [t for t in state.trade_history if t.get("session_start") == start_time]

    coin_summary = {}
    for t in session_trades:
        sym = t.get("symbol", "?")
        if sym not in coin_summary:
            coin_summary[sym] = {"trades": 0, "pnl": 0, "wins": 0}
        coin_summary[sym]["trades"] += 1
        coin_summary[sym]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            coin_summary[sym]["wins"] += 1

    lines = [
        f"📋 Последняя сессия\n",
        f"{'━' * 28}\n",
        f"⏱ Длительность: {dur_str}\n",
        f"💰 Баланс: ${start_bal:,.2f} → ${end_bal:,.2f}\n",
        f"{result_icon} Результат: {'+' if pnl >= 0 else ''}{pnl:,.2f}$ ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)\n",
        f"📊 Сделок: {trades_count}\n",
        f"\n{'━' * 28}\n",
        f"📊 По монетам:\n",
    ]

    if coin_summary:
        for sym, data in sorted(coin_summary.items(), key=lambda x: x[1]["pnl"], reverse=True):
            coin_name = sym.replace("USDT", "")
            wr = round(data["wins"] / max(data["trades"], 1) * 100, 1)
            p_icon = "🟩" if data["pnl"] >= 0 else "🟥"
            lines.append(
                f"  {p_icon} {coin_name}: {data['trades']} сделок | "
                f"WR {wr:.0f}% | {'+' if data['pnl'] >= 0 else ''}{data['pnl']:,.2f}$\n"
            )
    else:
        lines.append("  Нет данных по сделкам\n")

    return "".join(lines)


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


def format_trade_notification(action, data):
    if action == "open":
        direction = data.get("direction_str", "?")
        icon = "🔵" if direction == "LONG" else "🔴"
        symbol = data.get("symbol", "?").replace("USDT", "")
        price = data.get("entry_price", 0)
        size = data.get("size_usd", 0)
        lev = data.get("leverage", 1)
        conf = data.get("confidence", 0)

        return (
            f"{icon} ОТКРЫТА {direction} {symbol}/USDT\n"
            f"💲 Цена: ${price:,.4f}\n"
            f"📏 Размер: ${size:.2f} ({lev}x)\n"
            f"🎯 Уверенность: {conf:.1%}"
        )

    elif action == "close":
        symbol = data.get("symbol", "?").replace("USDT", "")
        pnl = data.get("pnl", 0)
        reason = data.get("reason", "?")
        pnl_icon = "🟩" if pnl >= 0 else "🟥"

        reason_map = {
            "TP": "🎯 Take Profit",
            "SL": "🛑 Stop Loss",
            "TRAILING_SL": "📐 Trailing Stop",
            "SESSION_END": "⏹ Конец сессии",
        }
        reason_str = reason_map.get(reason, reason)

        return (
            f"{pnl_icon} ЗАКРЫТА {symbol}/USDT\n"
            f"📋 Причина: {reason_str}\n"
            f"💰 P&L: {'+' if pnl >= 0 else ''}{pnl:,.2f}$"
        )

    return ""
