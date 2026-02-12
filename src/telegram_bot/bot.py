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
    session_detail_keyboard, confirm_reset_keyboard,
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

    elif data == "reset_balance":
        if state.trading_active:
            await _safe_edit(query, "\u274c \u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0435 \u0442\u043e\u0440\u0433\u043e\u0432\u043b\u044e!", back_keyboard())
            return
        text = (
            "\U0001f504 \u0421\u0431\u0440\u043e\u0441 \u0431\u0430\u043b\u0430\u043d\u0441\u0430\n\n"
            f"\U0001f4b0 \u0422\u0435\u043a\u0443\u0449\u0438\u0439: ${state.demo_balance:,.2f}\n"
            "\U0001f195 \u041d\u043e\u0432\u044b\u0439: $500.00\n\n"
            "\u26a0\ufe0f \u0421\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0430 \u0431\u0443\u0434\u0435\u0442 \u0441\u0431\u0440\u043e\u0448\u0435\u043d\u0430.\n"
            "\u0418\u0441\u0442\u043e\u0440\u0438\u044f \u0441\u0435\u0441\u0441\u0438\u0439 \u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u0441\u044f."
        )
        await _safe_edit(query, text, confirm_reset_keyboard())

    elif data == "confirm_reset":
        state.reset_balance()
        await refresh_main(query, state)

    elif data.startswith("session_detail_"):
        parts = data.split("_")
        session_num = int(parts[2])
        trade_page = int(parts[3]) if len(parts) > 3 else 0
        state.set("current_screen", "session_detail")
        session_data = state.get_session(session_num)
        text, total_pages = format_session_detail(session_data, trade_page)
        kb = session_detail_keyboard(session_num, trade_page, total_pages)
        await _safe_edit(query, text, kb)

    elif data == "noop":
        pass

    else:
        logger.warning(f"Unknown callback: {data}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)
    engine = get_engine(user_id)

    trade_status = "\U0001f7e2 \u0410\u043a\u0442\u0438\u0432\u043d\u0430" if state.trading_active else "\U0001f534 \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0430"
    engine_status = "\U0001f7e2 \u0437\u0430\u043f\u0443\u0449\u0435\u043d" if (engine and engine._running) else "\U0001f534 \u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d"
    lines = [
        f"\U0001f4ca \u0421\u0442\u0430\u0442\u0443\u0441 \u0431\u043e\u0442\u0430\n",
        f"\u0422\u043e\u0440\u0433\u043e\u0432\u043b\u044f: {trade_status}\n",
        f"\u0414\u0432\u0438\u0436\u043e\u043a: {engine_status}\n",
        f"\U0001f4b0 \u0411\u0430\u043b\u0430\u043d\u0441: ${state.demo_balance:,.2f}\n",
        f"\U0001f4ca \u041f\u043e\u0437\u0438\u0446\u0438\u0439: {len(state.open_positions)}\n",
        f"\U0001f4cb \u0421\u0434\u0435\u043b\u043e\u043a: {state.get('stats', {}).get('total_trades', 0)}\n",
        f"\U0001f3c6 \u0412\u0438\u043d\u0440\u0435\u0439\u0442: {state.get_win_rate():.1f}%\n",
        f"#\ufe0f\u20e3 \u0421\u0435\u0441\u0441\u0438\u044f: #{state.session_counter}\n",
    ]

    await update.message.reply_text("".join(lines))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "\U0001f916 AI Rvanka \u0422\u0440\u0435\u0439\u0434\u0438\u043d\u0433 \u0411\u043e\u0442\n\n"
        "\u041a\u043e\u043c\u0430\u043d\u0434\u044b:\n"
        "/start \u2014 \u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e\n"
        "/status \u2014 \u0421\u0442\u0430\u0442\u0443\u0441 \u0431\u043e\u0442\u0430\n"
        "/help \u2014 \u041f\u043e\u043c\u043e\u0449\u044c\n\n"
        "\U0001f4b9 6 \u043c\u043e\u043d\u0435\u0442: BTC, ETH, SOL, XRP, BNB, DOGE\n"
        "\U0001f9e0 LightGBM ML \u043c\u043e\u0434\u0435\u043b\u0438\n"
        "\U0001f4ca 179 \u0442\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0438\u043d\u0434\u0438\u043a\u0430\u0442\u043e\u0440\u043e\u0432\n"
        "\u23f0 5 \u0442\u0430\u0439\u043c\u0444\u0440\u0435\u0439\u043c\u043e\u0432 (1\u043c, 5\u043c, 15\u043c, 1\u0447, 4\u0447)\n"
        "\U0001f3af \u0414\u0438\u043d\u0430\u043c\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0422\u041f/\u0421\u041b + \u0422\u0440\u0435\u0439\u043b\u0438\u043d\u0433 \u0421\u0442\u043e\u043f\n"
        "\u2699\ufe0f \u0414\u0438\u043d\u0430\u043c\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u043f\u043b\u0435\u0447\u043e + \u041a\u043e\u043d\u0442\u0440\u043e\u043b\u044c \u0440\u0438\u0441\u043a\u0430"
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
