from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_keyboard(state):
    trading = state.trading_active

    if trading:
        trade_row = [InlineKeyboardButton("\u23f9 \u0421\u0442\u043e\u043f \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u0438", callback_data="stop_trading")]
    else:
        trade_row = [InlineKeyboardButton("\u25b6\ufe0f \u0421\u0442\u0430\u0440\u0442 \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u0438", callback_data="start_trading")]

    keyboard = [
        trade_row,
        [InlineKeyboardButton("\U0001f4cb \u0418\u0441\u0442\u043e\u0440\u0438\u044f \u0441\u0435\u0441\u0441\u0438\u0439", callback_data="sessions_list")],
    ]

    session_num = state.session_counter
    if session_num > 0:
        keyboard.append([InlineKeyboardButton(f"\U0001f4ca \u0421\u0435\u0441\u0441\u0438\u044f #{session_num}", callback_data=f"session_detail_{session_num}_0")])

    if not trading:
        keyboard.append([
            InlineKeyboardButton("\U0001f504 \u0421\u0431\u0440\u043e\u0441 \u0431\u0430\u043b\u0430\u043d\u0441\u0430", callback_data="reset_balance"),
            InlineKeyboardButton("\U0001f4b0 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c \u0431\u0430\u043b\u0430\u043d\u0441", callback_data="set_balance"),
        ])
        keyboard.append([
            InlineKeyboardButton("\U0001f4ca \u041c\u0430\u043a\u0441 \u043f\u043e\u0437\u0438\u0446\u0438\u0439", callback_data="set_positions"),
        ])

    return InlineKeyboardMarkup(keyboard)


def back_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434", callback_data="back_main")]
    ])


def session_detail_keyboard(session_num, trade_page=0, total_trade_pages=1):
    keyboard = []
    if total_trade_pages > 1:
        nav = []
        if trade_page > 0:
            nav.append(InlineKeyboardButton("\u25c0\ufe0f", callback_data=f"session_detail_{session_num}_{trade_page - 1}"))
        nav.append(InlineKeyboardButton(f"\U0001f4c4 {trade_page + 1}/{total_trade_pages}", callback_data="noop"))
        if trade_page < total_trade_pages - 1:
            nav.append(InlineKeyboardButton("\u25b6\ufe0f", callback_data=f"session_detail_{session_num}_{trade_page + 1}"))
        keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434", callback_data="back_main")])
    return InlineKeyboardMarkup(keyboard)


def sessions_list_keyboard(sessions, page=0, per_page=5):
    keyboard = []
    start = page * per_page
    end = min(start + per_page, len(sessions))

    for s in reversed(sessions[start:end]):
        num = s.get("session_number", "?")
        pnl = s.get("pnl", 0)
        icon = "\U0001f7e9" if pnl >= 0 else "\U0001f7e5"
        sign = "+" if pnl >= 0 else ""
        trades = s.get("total_trades", 0)
        wr = s.get("win_rate", 0)
        keyboard.append([InlineKeyboardButton(
            f"{icon} #{num} | {sign}{pnl:,.2f}$ | {trades} \u0441\u0434. | {wr:.0f}%",
            callback_data=f"session_detail_{num}_0",
        )])

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("\u25c0\ufe0f", callback_data=f"sessions_page_{page - 1}"))
    total_pages = max(1, (len(sessions) + per_page - 1) // per_page)
    nav_row.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if end < len(sessions):
        nav_row.append(InlineKeyboardButton("\u25b6\ufe0f", callback_data=f"sessions_page_{page + 1}"))
    if nav_row:
        keyboard.append(nav_row)

    keyboard.append([InlineKeyboardButton("\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434", callback_data="back_main")])
    return InlineKeyboardMarkup(keyboard)


def confirm_start_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 \u0414\u0430, \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c!", callback_data="confirm_start"),
            InlineKeyboardButton("\u274c \u041e\u0442\u043c\u0435\u043d\u0430", callback_data="back_main"),
        ]
    ])


def confirm_stop_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 \u0414\u0430, \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c!", callback_data="confirm_stop"),
            InlineKeyboardButton("\u274c \u041e\u0442\u043c\u0435\u043d\u0430", callback_data="back_main"),
        ]
    ])


def set_balance_keyboard():
    presets = [500, 1000, 5000, 10000]
    keyboard = []
    row = []
    for amt in presets:
        row.append(InlineKeyboardButton(f"${amt:,}", callback_data=f"setbal_{amt}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("\u270d\ufe0f \u0412\u0432\u0435\u0441\u0442\u0438 \u0441\u0432\u043e\u044e \u0441\u0443\u043c\u043c\u0443", callback_data="setbal_custom")])
    keyboard.append([InlineKeyboardButton("\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434", callback_data="back_main")])
    return InlineKeyboardMarkup(keyboard)


def set_positions_keyboard():
    presets = [2, 3, 4, 6, 7, 8]
    keyboard = []
    row = []
    for val in presets:
        row.append(InlineKeyboardButton(str(val), callback_data=f"setpos_{val}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("\u270d\ufe0f \u0412\u0432\u0435\u0441\u0442\u0438 \u0441\u0432\u043e\u0451 \u0447\u0438\u0441\u043b\u043e", callback_data="setpos_custom")])
    keyboard.append([InlineKeyboardButton("\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434", callback_data="back_main")])
    return InlineKeyboardMarkup(keyboard)


def confirm_reset_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 \u0414\u0430, \u0441\u0431\u0440\u043e\u0441\u0438\u0442\u044c!", callback_data="confirm_reset"),
            InlineKeyboardButton("\u274c \u041e\u0442\u043c\u0435\u043d\u0430", callback_data="back_main"),
        ]
    ])
