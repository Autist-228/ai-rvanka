import os
import sys
import logging
import asyncio

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.telegram_bot.state import UserState
from src.telegram_bot.engine import TradingEngine, ModelPredictor, PriceTracker
from src.telegram_bot.views import (
    format_main_view, format_sessions_list, format_session_detail,
)
from src.telegram_bot.keyboards import (
    main_keyboard, back_keyboard, sessions_list_keyboard,
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
        text = (
            f"\u25b6\ufe0f \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044e?\n\n"
            f"\U0001f4b0 \u0411\u0430\u043b\u0430\u043d\u0441: ${state.demo_balance:,.2f}\n"
            f"\U0001f4ca \u0421\u0435\u0441\u0441\u0438\u044f: #{state.session_counter + 1}\n"
        )
        await _safe_edit(query, text, confirm_start_keyboard())

    elif data == "confirm_start":
        global _predictor, _price_tracker

        if _predictor is None or not _predictor._loaded:
            await _safe_edit(query, "\u23f3 \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043c\u043e\u0434\u0435\u043b\u0435\u0439... (\u043f\u0435\u0440\u0432\u044b\u0439 \u0437\u0430\u043f\u0443\u0441\u043a ~30 \u0441\u0435\u043a)", None)
            _predictor = ModelPredictor("_prod")
            _predictor.load_models()
            _price_tracker = PriceTracker()

        engine = TradingEngine(state, _predictor, _price_tracker)

        async def on_update():
            screen = state.get("current_screen")
            if screen != "main":
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
        text = "\u23f9 \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044e?\n\n\u0412\u0441\u0435 \u043e\u0442\u043a\u0440\u044b\u0442\u044b\u0435 \u043f\u043e\u0437\u0438\u0446\u0438\u0438 \u0431\u0443\u0434\u0443\u0442 \u0437\u0430\u043a\u0440\u044b\u0442\u044b."
        await _safe_edit(query, text, confirm_stop_keyboard())

    elif data == "confirm_stop":
        engine = get_engine(user_id)
        if engine:
            await engine.stop()
            del _trading_engines[user_id]
        else:
            state.trading_active = False

        await refresh_main(query, state)

    elif data == "sessions_list":
        state.set("current_screen", "sessions_list")
        sessions = state.get_all_sessions()
        text = format_sessions_list(sessions)
        await _safe_edit(query, text, sessions_list_keyboard(sessions))

    elif data.startswith("sessions_page_"):
        page = int(data.split("_")[-1])
        sessions = state.get_all_sessions()
        text = format_sessions_list(sessions)
        await _safe_edit(query, text, sessions_list_keyboard(sessions, page=page))

    elif data.startswith("session_detail_"):
        session_num = int(data.split("_")[-1])
        state.set("current_screen", "session_detail")
        session_data = state.get_session(session_num)
        text = format_session_detail(session_data)
        await _safe_edit(query, text, back_keyboard())

    elif data == "noop":
        pass


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)
    engine = get_engine(user_id)

    lines = [
        f"\U0001f4ca \u0421\u0442\u0430\u0442\u0443\u0441 \u0431\u043e\u0442\u0430\n",
        f"\u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f: {'\u0410\u043a\u0442\u0438\u0432\u043d\u0430' if state.trading_active else '\u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0430'}\n",
        f"\u0411\u0430\u043b\u0430\u043d\u0441: ${state.demo_balance:,.2f}\n",
        f"\u041f\u043e\u0437\u0438\u0446\u0438\u0439: {len(state.open_positions)}\n",
        f"\u0421\u0434\u0435\u043b\u043e\u043a: {state.get('stats', {}).get('total_trades', 0)}\n",
        f"WR: {state.get_win_rate():.1f}%\n",
        f"\u0421\u0435\u0441\u0441\u0438\u044f: #{state.session_counter}\n",
    ]

    if engine and engine._running:
        lines.append(f"\u0414\u0432\u0438\u0436\u043e\u043a: \u0437\u0430\u043f\u0443\u0449\u0435\u043d\n")
    else:
        lines.append(f"\u0414\u0432\u0438\u0436\u043e\u043a: \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\n")

    await update.message.reply_text("".join(lines))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "\U0001f916 AI Rvanka Trading Bot\n\n"
        "\u041a\u043e\u043c\u0430\u043d\u0434\u044b:\n"
        "/start \u2014 \u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e\n"
        "/status \u2014 \u0421\u0442\u0430\u0442\u0443\u0441 \u0431\u043e\u0442\u0430\n"
        "/help \u2014 \u041f\u043e\u043c\u043e\u0449\u044c\n\n"
        "6 \u043c\u043e\u043d\u0435\u0442: BTC, ETH, SOL, XRP, BNB, DOGE\n"
        "LightGBM ML \u043c\u043e\u0434\u0435\u043b\u0438\n"
        "179 \u0442\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0438\u043d\u0434\u0438\u043a\u0430\u0442\u043e\u0440\u043e\u0432\n"
        "5 \u0442\u0430\u0439\u043c\u0444\u0440\u0435\u0439\u043c\u043e\u0432 (1m, 5m, 15m, 1h, 4h)\n"
        "ATR-based TP/SL + Trailing Stop\n"
        "Dynamic Leverage + Risk Controller"
    )
    await update.message.reply_text(text)


async def post_init(application):
    commands = [
        BotCommand("start", "\u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e"),
        BotCommand("status", "\u0421\u0442\u0430\u0442\u0443\u0441 \u0431\u043e\u0442\u0430"),
        BotCommand("help", "\u041f\u043e\u043c\u043e\u0449\u044c"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands set")


def run_bot():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
        return

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Starting bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
