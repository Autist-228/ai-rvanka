from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_keyboard(state):
    trading = state.trading_active

    if trading:
        trade_row = [InlineKeyboardButton("⏹ Стоп торговли", callback_data="stop_trading")]
    else:
        trade_row = [InlineKeyboardButton("▶️ Старт торговли", callback_data="start_trading")]

    keyboard = [
        trade_row,
        [InlineKeyboardButton("📋 История сессий", callback_data="sessions_list")],
    ]

    session_num = state.session_counter
    if session_num > 0:
        keyboard.append([InlineKeyboardButton(f"📊 Сессия #{session_num}", callback_data=f"session_detail_{session_num}")])

    return InlineKeyboardMarkup(keyboard)


def back_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")]
    ])


def sessions_list_keyboard(sessions, page=0, per_page=5):
    keyboard = []
    start = page * per_page
    end = min(start + per_page, len(sessions))

    for s in reversed(sessions[start:end]):
        num = s.get("session_number", "?")
        pnl = s.get("pnl", 0)
        icon = "🟩" if pnl >= 0 else "🟥"
        sign = "+" if pnl >= 0 else ""
        trades = s.get("total_trades", 0)
        keyboard.append([InlineKeyboardButton(
            f"{icon} #{num} | {sign}{pnl:,.2f}$ | {trades} сд.",
            callback_data=f"session_detail_{num}",
        )])

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️", callback_data=f"sessions_page_{page - 1}"))
    total_pages = max(1, (len(sessions) + per_page - 1) // per_page)
    nav_row.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if end < len(sessions):
        nav_row.append(InlineKeyboardButton("▶️", callback_data=f"sessions_page_{page + 1}"))
    if nav_row:
        keyboard.append(nav_row)

    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(keyboard)


def confirm_start_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, запустить!", callback_data="confirm_start"),
            InlineKeyboardButton("❌ Отмена", callback_data="back_main"),
        ]
    ])


def confirm_stop_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, остановить!", callback_data="confirm_stop"),
            InlineKeyboardButton("❌ Отмена", callback_data="back_main"),
        ]
    ])
