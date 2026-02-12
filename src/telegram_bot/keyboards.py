from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_keyboard(state):
    trading = state.trading_active
    is_demo = state.active_account == "demo"

    if trading:
        trade_row = [InlineKeyboardButton("⏹ Стоп торговли", callback_data="stop_trading")]
    else:
        trade_row = [InlineKeyboardButton("▶️ Старт торговли", callback_data="start_trading")]

    if is_demo:
        switch_text = "🔄 Переключить на Реал"
        switch_data = "switch_real"
    else:
        switch_text = "🔄 Переключить на Демо"
        switch_data = "switch_demo"

    switch_btn = InlineKeyboardButton(switch_text, callback_data=switch_data)

    if trading:
        switch_btn = InlineKeyboardButton(f"🔒 {switch_text.split(' ', 1)[1]}", callback_data="locked_switch")

    keyboard = [
        trade_row,
        [switch_btn],
        [InlineKeyboardButton("\U0001f4cb \u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u0441\u0435\u0441\u0441\u0438\u044f", callback_data="last_session")],
        [InlineKeyboardButton("\u2699\ufe0f \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438", callback_data="settings")],
    ]
    return InlineKeyboardMarkup(keyboard)


def back_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")]
    ])


def settings_keyboard():
    keyboard = [
        [InlineKeyboardButton("💰 Демо-баланс", callback_data="edit_demo_balance")],
        [InlineKeyboardButton("🎯 Уверенность сигнала", callback_data="edit_confidence_threshold")],
        [InlineKeyboardButton("⚡ Макс. leverage", callback_data="edit_max_leverage")],
        [InlineKeyboardButton("📏 Макс. позиция $", callback_data="edit_max_position_usd")],
        [InlineKeyboardButton("📊 Макс. сделок", callback_data="edit_max_concurrent_total")],
        [InlineKeyboardButton("📉 Стоп при просадке", callback_data="edit_max_drawdown_pct")],
        [InlineKeyboardButton("⏸ Пауза после лузов", callback_data="edit_consecutive_sl_pause")],
        [InlineKeyboardButton("🎯 TP/SL по монетам", callback_data="edit_coin_tpsl")],
        [InlineKeyboardButton("🔄 Сбросить статистику", callback_data="reset_stats")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
    ]
    return InlineKeyboardMarkup(keyboard)


def confidence_keyboard():
    values = [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"{v:.0%}", callback_data=f"set_confidence_{v}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def leverage_keyboard():
    values = [1, 3, 5, 7, 10, 15, 20]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"{v}x", callback_data=f"set_leverage_{v}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def max_position_keyboard():
    values = [50, 100, 150, 200, 300, 500]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"${v}", callback_data=f"set_max_pos_{v}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def max_concurrent_keyboard():
    values = [1, 2, 3, 4, 5, 6]
    row = [InlineKeyboardButton(str(v), callback_data=f"set_concurrent_{v}") for v in values]
    return InlineKeyboardMarkup([
        row,
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")],
    ])


def drawdown_keyboard():
    values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"{v:.0%}", callback_data=f"set_drawdown_{v}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def pause_keyboard():
    values = [2, 3, 4, 5]
    row = [InlineKeyboardButton(str(v), callback_data=f"set_pause_{v}") for v in values]
    return InlineKeyboardMarkup([
        row,
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")],
    ])


def demo_balance_keyboard():
    values = [100, 250, 500, 1000, 2500, 5000]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"${v}", callback_data=f"set_demo_bal_{v}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def coin_tpsl_keyboard():
    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
    rows = []
    for coin in coins:
        name = coin.replace("USDT", "")
        rows.append([InlineKeyboardButton(f"🎯 {name}", callback_data=f"coin_tpsl_{coin}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


def coin_tp_edit_keyboard(symbol):
    values = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"{v*100:.1f}%", callback_data=f"set_tp_{symbol}_{v}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="edit_coin_tpsl")])
    return InlineKeyboardMarkup(rows)


def coin_sl_edit_keyboard(symbol):
    values = [0.005, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03]
    rows = []
    row = []
    for v in values:
        row.append(InlineKeyboardButton(f"{v*100:.1f}%", callback_data=f"set_sl_{symbol}_{v}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="edit_coin_tpsl")])
    return InlineKeyboardMarkup(rows)


def coin_tpsl_detail_keyboard(symbol):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎯 Изменить TP", callback_data=f"edit_tp_{symbol}"),
            InlineKeyboardButton("🛑 Изменить SL", callback_data=f"edit_sl_{symbol}"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="edit_coin_tpsl")],
    ])


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
