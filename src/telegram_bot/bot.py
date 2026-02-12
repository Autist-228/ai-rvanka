import os
import sys
import logging
import asyncio

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters,
)
from telegram.constants import ParseMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.telegram_bot.state import UserState
from src.telegram_bot.engine import TradingEngine, ModelPredictor, PriceTracker
from src.telegram_bot.views import (
    format_main_view, format_last_session,
    format_settings, format_setting_edit, format_coin_tpsl_edit,
)
from src.telegram_bot.keyboards import (
    main_keyboard, back_keyboard, settings_keyboard,
    confidence_keyboard, leverage_keyboard, max_position_keyboard,
    max_concurrent_keyboard, drawdown_keyboard, pause_keyboard,
    demo_balance_keyboard, coin_tpsl_keyboard, coin_tpsl_detail_keyboard,
    coin_tp_edit_keyboard, coin_sl_edit_keyboard,
    confirm_start_keyboard, confirm_stop_keyboard,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

_user_states = {}
_trading_engines = {}
_predictor = None
_price_tracker = None


def get_state(user_id):
    if user_id not in _user_states:
        _user_states[user_id] = UserState(user_id)
    return _user_states[user_id]


def get_engine(user_id):
    return _trading_engines.get(user_id)


async def _safe_edit(query, text, reply_markup=None):
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        await query.edit_message_text(
            text=text,
            reply_markup=reply_markup,
        )
    except Exception as e:
        err = str(e)
        if "Message is not modified" not in err:
            logger.error(f"Edit error: {e}")
            if "Bad Request" in err:
                try:
                    await query.edit_message_text(
                        text="⚠️ Ошибка отображения. Нажмите 🔄",
                        reply_markup=reply_markup,
                    )
                except Exception:
                    pass


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)

    text = format_main_view(state)
    msg = await update.message.reply_text(
        text=text,
        reply_markup=main_keyboard(state),
    )

    state.set("main_message_id", msg.message_id)
    state.set("main_chat_id", msg.chat_id)
    state.set("current_screen", "main")


async def refresh_main(query, state):
    text = format_main_view(state)
    await _safe_edit(query, text, main_keyboard(state))


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    state = get_state(user_id)
    data = query.data

    if data == "refresh" or data == "back_main":
        state.set("current_screen", "main")
        await refresh_main(query, state)

    elif data == "start_trading":
        if state.active_account == "real" and not state.get("bybit_api_key"):
            await _safe_edit(
                query,
                "❌ Для реального счёта нужен API ключ Bybit!\n\n"
                "Отправьте ключи командой:\n"
                "/set_api KEY SECRET",
                back_keyboard(),
            )
            return
        text = (
            f"▶️ Запустить торговлю?\n\n"
            f"Счёт: {'🟢 Демо' if state.active_account == 'demo' else '🔵 Реал'}\n"
            f"Баланс: ${state.demo_balance:,.2f}\n"
            f"Уверенность: {state.settings.get('confidence_threshold', 0.65):.0%}\n"
            f"Макс. leverage: {state.settings.get('max_leverage', 20)}x\n"
            f"Макс. позиций: {state.settings.get('max_concurrent_total', 3)}\n"
        )
        await _safe_edit(query, text, confirm_start_keyboard())

    elif data == "confirm_start":
        global _predictor, _price_tracker

        if _predictor is None or not _predictor._loaded:
            await _safe_edit(query, "⏳ Загрузка моделей... (первый запуск ~30 сек)", None)
            _predictor = ModelPredictor("_prod")
            _predictor.load_models()
            _price_tracker = PriceTracker()

        engine = TradingEngine(state, _predictor, _price_tracker)

        async def on_update():
            screen = state.get("current_screen")
            if screen != "main":
                logger.debug(f"on_update skip: screen={screen}")
                return
            msg_id = state.get("main_message_id")
            chat_id = state.get("main_chat_id")
            if msg_id and chat_id:
                try:
                    text = format_main_view(state)
                    if len(text) > 4096:
                        text = text[:4090] + "\n..."
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text=text,
                        reply_markup=main_keyboard(state),
                    )
                except Exception as e:
                    err_str = str(e)
                    if "not modified" not in err_str.lower():
                        logger.error(f"on_update edit error: {e}")

        engine.set_callbacks(on_trade=None, on_update=on_update)
        _trading_engines[user_id] = engine
        await engine.start()

        await refresh_main(query, state)

    elif data == "stop_trading":
        text = "⏹ Остановить торговлю?\n\nВсе открытые позиции будут закрыты."
        await _safe_edit(query, text, confirm_stop_keyboard())

    elif data == "confirm_stop":
        engine = get_engine(user_id)
        if engine:
            await engine.stop()
            del _trading_engines[user_id]
        else:
            state.trading_active = False

        await refresh_main(query, state)

    elif data == "switch_demo":
        if state.trading_active:
            await query.answer("Сначала остановите торговлю!", show_alert=True)
            return
        state.active_account = "demo"
        await refresh_main(query, state)

    elif data == "switch_real":
        if state.trading_active:
            await query.answer("Сначала остановите торговлю!", show_alert=True)
            return
        state.active_account = "real"
        await refresh_main(query, state)

    elif data == "locked_switch":
        await query.answer("Сначала остановите торговлю!", show_alert=True)

    elif data == "open_positions":
        state.set("current_screen", "main")
        await refresh_main(query, state)

    elif data == "last_session":
        state.set("current_screen", "last_session")
        text = format_last_session(state)
        await _safe_edit(query, text, back_keyboard())

    elif data == "settings":
        state.set("current_screen", "settings")
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_confidence_threshold":
        current = state.settings.get("confidence_threshold", 0.65)
        text = format_setting_edit("confidence_threshold", current)
        await _safe_edit(query, text, confidence_keyboard())

    elif data.startswith("set_confidence_"):
        val = float(data.split("_")[-1])
        state.update_setting("confidence_threshold", val)
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_max_leverage":
        current = state.settings.get("max_leverage", 20)
        text = format_setting_edit("max_leverage", current)
        await _safe_edit(query, text, leverage_keyboard())

    elif data.startswith("set_leverage_"):
        val = int(data.split("_")[-1])
        state.update_setting("max_leverage", val)
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_max_position_usd":
        current = state.settings.get("max_position_usd", 300)
        text = format_setting_edit("max_position_usd", current)
        await _safe_edit(query, text, max_position_keyboard())

    elif data.startswith("set_max_pos_"):
        val = int(data.split("_")[-1])
        state.update_setting("max_position_usd", float(val))
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_max_concurrent_total":
        current = state.settings.get("max_concurrent_total", 3)
        text = format_setting_edit("max_concurrent_total", current)
        await _safe_edit(query, text, max_concurrent_keyboard())

    elif data.startswith("set_concurrent_"):
        val = int(data.split("_")[-1])
        state.update_setting("max_concurrent_total", val)
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_max_drawdown_pct":
        current = state.settings.get("max_drawdown_pct", 0.15)
        text = format_setting_edit("max_drawdown_pct", current)
        await _safe_edit(query, text, drawdown_keyboard())

    elif data.startswith("set_drawdown_"):
        val = float(data.split("_")[-1])
        state.update_setting("max_drawdown_pct", val)
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_consecutive_sl_pause":
        current = state.settings.get("consecutive_sl_pause", 3)
        text = format_setting_edit("consecutive_sl_pause", current)
        await _safe_edit(query, text, pause_keyboard())

    elif data.startswith("set_pause_"):
        val = int(data.split("_")[-1])
        state.update_setting("consecutive_sl_pause", val)
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_demo_balance":
        text = format_setting_edit("demo_balance", state.demo_balance)
        await _safe_edit(query, text, demo_balance_keyboard())

    elif data.startswith("set_demo_bal_"):
        val = int(data.split("_")[-1])
        if state.trading_active:
            await query.answer("Сначала остановите торговлю!", show_alert=True)
            return
        state.demo_balance = float(val)
        state.set("demo_balance_initial", float(val))
        text = format_settings(state)
        await _safe_edit(query, text, settings_keyboard())

    elif data == "edit_coin_tpsl":
        text = "🎯 TP/SL по монетам\n\nВыберите монету:"
        await _safe_edit(query, text, coin_tpsl_keyboard())

    elif data.startswith("coin_tpsl_"):
        symbol = data.replace("coin_tpsl_", "")
        coin_tpsl = state.settings.get("coin_tp_sl", {}).get(symbol, {})
        tp = coin_tpsl.get("tp_pct", 0.04)
        sl = coin_tpsl.get("sl_pct", 0.02)
        text = format_coin_tpsl_edit(symbol, tp, sl)
        await _safe_edit(query, text, coin_tpsl_detail_keyboard(symbol))

    elif data.startswith("edit_tp_"):
        symbol = data.replace("edit_tp_", "")
        await _safe_edit(query, f"🎯 Take Profit для {symbol.replace('USDT', '')}\n\nВыберите значение:", coin_tp_edit_keyboard(symbol))

    elif data.startswith("edit_sl_"):
        symbol = data.replace("edit_sl_", "")
        await _safe_edit(query, f"🛑 Stop Loss для {symbol.replace('USDT', '')}\n\nВыберите значение:", coin_sl_edit_keyboard(symbol))

    elif data.startswith("set_tp_"):
        parts = data.split("_")
        symbol = parts[2]
        val = float(parts[3])
        coin_tpsl = state.settings.get("coin_tp_sl", {})
        if symbol not in coin_tpsl:
            coin_tpsl[symbol] = {"tp_pct": 0.04, "sl_pct": 0.02}
        coin_tpsl[symbol]["tp_pct"] = val
        state.update_setting("coin_tp_sl", coin_tpsl)
        tp = coin_tpsl[symbol]["tp_pct"]
        sl = coin_tpsl[symbol]["sl_pct"]
        text = format_coin_tpsl_edit(symbol, tp, sl)
        await _safe_edit(query, text, coin_tpsl_detail_keyboard(symbol))

    elif data.startswith("set_sl_"):
        parts = data.split("_")
        symbol = parts[2]
        val = float(parts[3])
        coin_tpsl = state.settings.get("coin_tp_sl", {})
        if symbol not in coin_tpsl:
            coin_tpsl[symbol] = {"tp_pct": 0.04, "sl_pct": 0.02}
        coin_tpsl[symbol]["sl_pct"] = val
        state.update_setting("coin_tp_sl", coin_tpsl)
        tp = coin_tpsl[symbol]["tp_pct"]
        sl = coin_tpsl[symbol]["sl_pct"]
        text = format_coin_tpsl_edit(symbol, tp, sl)
        await _safe_edit(query, text, coin_tpsl_detail_keyboard(symbol))


async def set_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)

    if len(context.args) != 2:
        await update.message.reply_text(
            "Использование: /set_api API_KEY API_SECRET\n\n"
            "⚠️ Совет: удалите это сообщение после ввода!"
        )
        return

    api_key, api_secret = context.args
    state.set("bybit_api_key", api_key)
    state.set("bybit_api_secret", api_secret)

    try:
        await update.message.delete()
    except Exception:
        pass

    await update.effective_chat.send_message(
        "✅ API ключи Bybit сохранены!\n\n"
        "⚠️ Сообщение с ключами было удалено для безопасности."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)
    engine = get_engine(user_id)

    lines = [
        f"📊 Статус бота\n",
        f"Счёт: {'Демо' if state.active_account == 'demo' else 'Реал'}\n",
        f"Торговля: {'Активна' if state.trading_active else 'Остановлена'}\n",
        f"Баланс: ${state.demo_balance:,.2f}\n",
        f"Открытых позиций: {len(state.open_positions)}\n",
        f"Всего сделок: {state.get('stats', {}).get('total_trades', 0)}\n",
        f"Win Rate: {state.get_win_rate():.1f}%\n",
    ]

    if engine and engine._running:
        lines.append(f"Движок: запущен\n")
    else:
        lines.append(f"Движок: остановлен\n")

    await update.message.reply_text("".join(lines))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 AI Rvanka Trading Bot\n\n"
        "Команды:\n"
        "/start — Главное меню\n"
        "/status — Статус бота\n"
        "/set_api KEY SECRET — Установить API Bybit\n"
        "/help — Помощь\n\n"
        "Бот использует 12 ML моделей (LightGBM + XGBoost)\n"
        "обученных на 2 годах данных для 6 монет:\n"
        "BTC, ETH, SOL, XRP, BNB, DOGE\n\n"
        "179 технических индикаторов\n"
        "5 таймфреймов (1m, 5m, 15m, 1h, 4h)\n"
        "Ensemble голосование (оба алгоритма должны согласиться)\n"
        "Динамический leverage на основе уверенности\n"
        "Per-coin TP/SL + trailing stop\n"
        "8+ уровней защиты капитала"
    )
    await update.message.reply_text(text)


async def post_init(application):
    commands = [
        BotCommand("start", "Главное меню"),
        BotCommand("status", "Статус бота"),
        BotCommand("help", "Помощь"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands set")


def run_bot():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
        print("Example: export TELEGRAM_BOT_TOKEN='your_token_here'")
        return

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("set_api", set_api_command))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Starting bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
