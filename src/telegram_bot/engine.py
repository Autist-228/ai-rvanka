import os
import sys
import time
import uuid
import asyncio
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.settings import MODELS_DIR, SYMBOLS, PRIMARY_TF, ALL_TIMEFRAMES, HIGHER_TF_FEATURES, LOWER_TF_FEATURES
from src.feature_engine import (
    compute_indicators_for_tf, add_cross_tf_ratios,
    add_cross_coin_features, get_feature_cols,
)
from src.portfolio_manager import MetaAllocator, RiskController, PORTFOLIO_CONFIG

logger = logging.getLogger(__name__)

PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")

TRAIL_ACTIVATION_PCT = PORTFOLIO_CONFIG["risk_controller"]["trailing_activation_pct"]
TRAIL_STEP_PCT = PORTFOLIO_CONFIG["risk_controller"]["trailing_step_pct"]


def _get_proxies():
    if not PROXY_HOST or not PROXY_PORT:
        return {}
    if PROXY_USER:
        url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
    else:
        url = f"http://{PROXY_HOST}:{PROXY_PORT}"
    return {"http": url, "https": url}


class ModelPredictor:
    def __init__(self, model_suffix="_prod"):
        self.models = {}
        self.feature_cols = {}
        self.model_suffix = model_suffix
        self._loaded = False

    def load_models(self):
        for symbol in SYMBOLS:
            lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm{self.model_suffix}.pkl")
            fc_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols{self.model_suffix}.pkl")

            if not os.path.exists(lgb_path) or not os.path.exists(fc_path):
                logger.warning(f"Missing model for {symbol}")
                continue

            self.models[symbol] = joblib.load(lgb_path)
            self.feature_cols[symbol] = joblib.load(fc_path)
            logger.info(f"Loaded model for {symbol}")

        self._loaded = True
        logger.info(f"Loaded {len(self.models)} coin models")

    def predict(self, symbol, features_df):
        if symbol not in self.models:
            return None

        model = self.models[symbol]
        trained_fc = self.feature_cols[symbol]

        df = features_df.copy()
        missing = [c for c in trained_fc if c not in df.columns]
        for col in missing:
            df[col] = 0
        df[trained_fc] = df[trained_fc].fillna(0)

        X = df[trained_fc].values[-1:].reshape(1, -1)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label_map = {0: 0, 1: 1, 2: -1}
        direction = label_map.get(pred, 0)
        confidence = proba[pred]

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
        }


class PriceTracker:
    def __init__(self):
        self._prices = {}
        self._kline_cache = {}
        self._highs = {}
        self._lows = {}

    async def fetch_current_prices(self):
        import aiohttp
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear"}
        proxies = _get_proxies()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, proxy=proxies.get("https"), timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        for item in data["result"]["list"]:
                            sym = item["symbol"]
                            if sym in SYMBOLS:
                                self._prices[sym] = {
                                    "last": float(item["lastPrice"]),
                                    "high24h": float(item.get("highPrice24h", 0)),
                                    "low24h": float(item.get("lowPrice24h", 0)),
                                    "volume24h": float(item.get("volume24h", 0)),
                                    "change24h": float(item.get("price24hPcnt", 0)) * 100,
                                    "timestamp": time.time(),
                                }
        except Exception as e:
            logger.error(f"Price fetch error: {e}")

    async def fetch_klines(self, symbol, interval="15", limit=300):
        import aiohttp
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        proxies = _get_proxies()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, proxy=proxies.get("https"), timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        rows = []
                        for k in reversed(data["result"]["list"]):
                            rows.append({
                                "timestamp": int(k[0]),
                                "datetime": pd.Timestamp(int(k[0]), unit="ms"),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                            })
                        df = pd.DataFrame(rows)
                        self._kline_cache[f"{symbol}_{interval}"] = df
                        return df
        except Exception as e:
            logger.error(f"Kline fetch error {symbol} {interval}: {e}")
        return self._kline_cache.get(f"{symbol}_{interval}")

    async def fetch_recent_high_low(self, symbol, interval="1", limit=4):
        import aiohttp
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        proxies = _get_proxies()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, proxy=proxies.get("https"), timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    if data.get("retCode") == 0:
                        highs = [float(k[2]) for k in data["result"]["list"]]
                        lows = [float(k[3]) for k in data["result"]["list"]]
                        self._highs[symbol] = max(highs) if highs else 0
                        self._lows[symbol] = min(lows) if lows else 0
                        return self._highs[symbol], self._lows[symbol]
        except Exception as e:
            logger.error(f"High/Low fetch error {symbol}: {e}")
        return self._highs.get(symbol, 0), self._lows.get(symbol, 0)

    def get_price(self, symbol):
        return self._prices.get(symbol, {}).get("last")

    def get_all_prices(self):
        return dict(self._prices)


def _compute_atr_tpsl(symbol, entry_price, direction, atr_value):
    config = PORTFOLIO_CONFIG
    trading = config["trading"]
    coin_tpsl = config.get("coin_tp_sl", {}).get(symbol, {})

    base_tp = coin_tpsl.get("tp_pct", trading["base_tp_pct"])
    base_sl = coin_tpsl.get("sl_pct", trading["base_sl_pct"])
    atr_tp_m = coin_tpsl.get("atr_tp_mult", 2.5)
    atr_sl_m = coin_tpsl.get("atr_sl_mult", 1.2)

    if atr_value and atr_value > 0:
        atr_pct = atr_value / entry_price
        tp_mult = max(base_tp, atr_pct * atr_tp_m)
        sl_mult = max(base_sl, atr_pct * atr_sl_m)
    else:
        tp_mult = base_tp
        sl_mult = base_sl

    if direction == 1:
        tp_price = entry_price * (1 + tp_mult)
        sl_price = entry_price * (1 - sl_mult)
    else:
        tp_price = entry_price * (1 - tp_mult)
        sl_price = entry_price * (1 + sl_mult)

    return round(tp_price, 6), round(sl_price, 6), round(tp_mult, 6), round(sl_mult, 6)


def _compute_dynamic_leverage(confidence, volatility, leverage_mult=1.0):
    trading = PORTFOLIO_CONFIG["trading"]
    conf_threshold = trading["confidence_threshold"]
    conf_ratio = (confidence - conf_threshold) / (1.0 - conf_threshold)
    conf_ratio = min(max(conf_ratio, 0.0), 1.0)

    if volatility is not None and volatility > 0:
        vol_factor = min(1.0, 0.02 / (volatility + 1e-10))
    else:
        vol_factor = 0.5

    combined = conf_ratio * 0.7 + vol_factor * 0.3
    max_lev = trading["max_leverage"]
    min_lev = trading["min_leverage"]

    leverage = min_lev + combined * (max_lev - min_lev)
    leverage = leverage * leverage_mult
    leverage = max(min_lev, min(max_lev, int(leverage)))
    return leverage


def _detect_regime(features_df):
    if features_df is None or len(features_df) == 0:
        return "normal"
    last = features_df.iloc[-1]
    adx_val = last.get("adx", 25) if "adx" in features_df.columns else 25
    vol_val = last.get("volatility_20", 0.02) if "volatility_20" in features_df.columns else 0.02
    if isinstance(adx_val, (int, float)) and isinstance(vol_val, (int, float)):
        if adx_val < 15 and vol_val > 0.04:
            return "crash"
        elif adx_val > 30:
            return "trend"
    return "normal"


def _get_atr(features_df):
    if features_df is None or len(features_df) == 0:
        return 0
    if "atr_14" in features_df.columns:
        val = features_df.iloc[-1].get("atr_14", 0)
        if isinstance(val, (int, float)) and not np.isnan(val):
            return val
    return 0


def _get_volatility(features_df):
    if features_df is None or len(features_df) == 0:
        return None
    if "volatility_20" in features_df.columns:
        val = features_df.iloc[-1].get("volatility_20", None)
        if isinstance(val, (int, float)) and not np.isnan(val):
            return val
    return None


class TradingEngine:
    def __init__(self, user_state, predictor, price_tracker):
        self.state = user_state
        self.predictor = predictor
        self.prices = price_tracker
        self._running = False
        self._task = None
        self._refresh_task = None
        self._on_trade_callback = None
        self._on_update_callback = None

        config = PORTFOLIO_CONFIG
        self.risk_ctrl = RiskController(config["risk_controller"])
        self.risk_ctrl.peak_balance = user_state.demo_balance
        self.allocator = MetaAllocator(config["allocator"])
        self._candle_idx = 0

    def set_callbacks(self, on_trade=None, on_update=None):
        self._on_trade_callback = on_trade
        self._on_update_callback = on_update

    async def start(self):
        if self._running:
            return
        self._running = True
        self.state.start_session()
        self._task = asyncio.create_task(self._signal_loop())
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(f"Trading started for user {self.state.user_id}")

    async def stop(self):
        self._running = False
        for task in [self._task, self._refresh_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self._close_all_positions("SESSION_END")
        self.state.stop_session()
        logger.info(f"Trading stopped for user {self.state.user_id}")

    async def _refresh_loop(self):
        logger.info("Refresh loop STARTED")
        while self._running:
            try:
                await self.prices.fetch_current_prices()
                self.state._live_prices = self.prices.get_all_prices()
                self.state._last_update_ts = time.time()
                await self._update_positions()

                if self._on_update_callback:
                    try:
                        await self._on_update_callback()
                    except Exception as cb_err:
                        logger.error(f"on_update callback error: {cb_err}")

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                logger.info("Refresh loop CANCELLED")
                break
            except Exception as e:
                logger.error(f"Refresh loop error: {e}")
                await asyncio.sleep(5)
        logger.info("Refresh loop ENDED")

    async def _signal_loop(self):
        logger.info("Signal loop STARTED")
        while self._running:
            try:
                await self._check_signals()
                self._candle_idx += 1
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Signal loop CANCELLED")
                break
            except Exception as e:
                logger.error(f"Signal loop error: {e}")
                await asyncio.sleep(30)

    async def _check_signals(self):
        if not self.state.trading_active:
            return

        config = PORTFOLIO_CONFIG
        trading = config["trading"]
        positions = self.state.open_positions
        max_concurrent = self.state.max_positions

        if len(positions) >= max_concurrent:
            return

        active_symbols = {p["symbol"] for p in positions}
        conf_threshold = trading["confidence_threshold"]
        logger.info(f"Signal check: {len(positions)}/{max_concurrent} pos, threshold={conf_threshold:.2f}")

        signals = {}

        for symbol in SYMBOLS:
            if symbol in active_symbols:
                continue

            try:
                klines = {}
                tf_intervals = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240"}
                fetch_tasks = []
                for tf_name, tf_int in tf_intervals.items():
                    fetch_tasks.append((tf_name, self.prices.fetch_klines(symbol, tf_int, 300)))
                results = await asyncio.gather(*[t[1] for t in fetch_tasks], return_exceptions=True)
                for (tf_name, _), result in zip(fetch_tasks, results):
                    if isinstance(result, Exception):
                        logger.warning(f"{symbol} {tf_name} kline error: {result}")
                    elif result is not None:
                        klines[tf_name] = result

                if PRIMARY_TF not in klines:
                    continue

                features_df = self._build_live_features(symbol, klines)
                if features_df is None or len(features_df) < 50:
                    continue

                prediction = self.predictor.predict(symbol, features_df)
                if prediction is None:
                    continue

                direction = prediction["direction"]
                confidence = prediction["confidence"]
                dir_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(direction, "?")
                logger.info(f"{symbol}: {dir_str} conf={confidence:.3f} (need>{conf_threshold:.2f})")

                regime = _detect_regime(features_df)
                atr = _get_atr(features_df)
                volatility = _get_volatility(features_df)

                signals[symbol] = {
                    "direction": direction,
                    "confidence": confidence,
                    "threshold": conf_threshold,
                    "candle_idx": self._candle_idx,
                    "regime": regime,
                    "volatility": volatility,
                    "atr": atr,
                    "close": self.prices.get_price(symbol) or 0,
                }

            except Exception as e:
                logger.error(f"Signal check error {symbol}: {e}")

        if not signals:
            return

        allocations = self.allocator.allocate(signals, self.risk_ctrl)
        if not allocations:
            return

        balance = self.state.demo_balance
        initial_balance = self.state.get("demo_balance_initial", 500.0)
        leverage_mult = self.risk_ctrl.get_leverage_multiplier(balance, initial_balance)

        for symbol, alloc_pct in allocations.items():
            sig = signals[symbol]
            if sig["direction"] == 0:
                continue
            if len(self.state.open_positions) >= max_concurrent:
                break

            max_pos_usd = config["allocator"].get("max_position_usd", 300.0)
            size_usd = min(balance * alloc_pct, max_pos_usd)
            if size_usd < 1:
                continue

            leverage = _compute_dynamic_leverage(sig["confidence"], sig["volatility"], leverage_mult)
            price = self.prices.get_price(symbol)
            if price is None or price <= 0:
                continue

            atr = sig["atr"]
            tp_price, sl_price, tp_mult, sl_mult = _compute_atr_tpsl(symbol, price, sig["direction"], atr)

            position = {
                "id": str(uuid.uuid4())[:8],
                "symbol": symbol,
                "direction": sig["direction"],
                "direction_str": "LONG" if sig["direction"] == 1 else "SHORT",
                "entry_price": price,
                "current_price": price,
                "size_usd": round(size_usd, 2),
                "leverage": leverage,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "original_sl_price": sl_price,
                "tp_pct": tp_mult,
                "sl_pct": sl_mult,
                "confidence": sig["confidence"],
                "regime": sig["regime"],
                "atr": atr,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "trailing_activated": False,
                "highest_price": price,
                "lowest_price": price,
                "open_time": datetime.utcnow().isoformat(),
                "session_id": self.state.current_session_id,
            }

            self.state.add_position(position)

            fee = size_usd * leverage * 0.00055
            self.state.demo_balance = self.state.demo_balance - fee

            logger.info(f"OPEN {position['direction_str']} {symbol} @ {price} | "
                         f"Size: ${size_usd:.2f} | Lev: {leverage}x | Conf: {sig['confidence']:.3f} | "
                         f"TP: {tp_price:.4f} | SL: {sl_price:.4f} | Regime: {sig['regime']}")

            if self._on_trade_callback:
                await self._on_trade_callback("open", position)

    def _build_live_features(self, symbol, klines):
        try:
            primary_df = klines.get(PRIMARY_TF)
            if primary_df is None:
                return None

            df = compute_indicators_for_tf(primary_df, PRIMARY_TF, is_primary=True)

            for tf in ["1h", "4h"]:
                if tf in klines:
                    htf = compute_indicators_for_tf(klines[tf], tf, is_primary=False)
                    htf_cols = ["timestamp"]
                    prefix = f"{tf}_"
                    for col in HIGHER_TF_FEATURES:
                        full_col = f"{prefix}{col}"
                        if full_col in htf.columns:
                            htf_cols.append(full_col)
                    htf_merge = htf[htf_cols].copy()
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        htf_merge.sort_values("timestamp"),
                        on="timestamp", direction="backward",
                    )

            for tf in ["1m", "5m"]:
                if tf in klines:
                    ltf = compute_indicators_for_tf(klines[tf], tf, is_primary=False)
                    ltf_cols = ["timestamp"]
                    prefix = f"{tf}_"
                    for col in LOWER_TF_FEATURES:
                        full_col = f"{prefix}{col}"
                        if full_col in ltf.columns:
                            ltf_cols.append(full_col)
                    ltf_merge = ltf[ltf_cols].copy()
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        ltf_merge.sort_values("timestamp"),
                        on="timestamp", direction="backward",
                    )

            df = add_cross_tf_ratios(df)

            if symbol != "BTCUSDT" and "15m" in klines:
                btc_klines = self.prices._kline_cache.get("BTCUSDT_15")
                if btc_klines is not None:
                    df = add_cross_coin_features(df, btc_klines)

            df = df.replace([np.inf, -np.inf], np.nan)
            return df

        except Exception as e:
            logger.error(f"Feature build error {symbol}: {e}")
            return None

    async def _update_positions(self):
        positions = self.state.open_positions
        if not positions:
            return

        updated = []
        for pos in positions:
            symbol = pos["symbol"]
            price = self.prices.get_price(symbol)
            if price is None:
                updated.append(pos)
                continue

            pos["current_price"] = price
            direction = pos["direction"]
            entry = pos["entry_price"]

            high, low = await self.prices.fetch_recent_high_low(symbol, "1", 4)
            if high == 0:
                high = price
            if low == 0:
                low = price

            if direction == 1:
                if high > pos.get("highest_price", entry):
                    pos["highest_price"] = high
                unrealized_pct = (pos["highest_price"] - entry) / entry
                if unrealized_pct >= TRAIL_ACTIVATION_PCT:
                    pos["trailing_activated"] = True
                    new_sl = pos["highest_price"] * (1 - TRAIL_STEP_PCT)
                    if new_sl > pos["sl_price"]:
                        old_sl = pos["sl_price"]
                        pos["sl_price"] = round(new_sl, 6)
                        logger.info(f"TRAILING SL {symbol}: {old_sl:.4f} -> {pos['sl_price']:.4f}")
            else:
                if low < pos.get("lowest_price", entry):
                    pos["lowest_price"] = low
                unrealized_pct = (entry - pos["lowest_price"]) / entry
                if unrealized_pct >= TRAIL_ACTIVATION_PCT:
                    pos["trailing_activated"] = True
                    new_sl = pos["lowest_price"] * (1 + TRAIL_STEP_PCT)
                    if new_sl < pos["sl_price"]:
                        old_sl = pos["sl_price"]
                        pos["sl_price"] = round(new_sl, 6)
                        logger.info(f"TRAILING SL {symbol}: {old_sl:.4f} -> {pos['sl_price']:.4f}")

            if direction == 1:
                price_change = (price - entry) / entry
            else:
                price_change = (entry - price) / entry

            trade_value = pos["size_usd"] * pos["leverage"]
            pos["pnl"] = round(trade_value * price_change, 2)
            pos["pnl_pct"] = round(price_change * 100, 2)

            should_close = False
            close_reason = None

            if direction == 1:
                if high >= pos["tp_price"]:
                    should_close = True
                    close_reason = "TP"
                    pos["current_price"] = pos["tp_price"]
                elif low <= pos["sl_price"]:
                    should_close = True
                    close_reason = "TRAILING_SL" if pos.get("trailing_activated") else "SL"
                    pos["current_price"] = pos["sl_price"]
            else:
                if low <= pos["tp_price"]:
                    should_close = True
                    close_reason = "TP"
                    pos["current_price"] = pos["tp_price"]
                elif high >= pos["sl_price"]:
                    should_close = True
                    close_reason = "TRAILING_SL" if pos.get("trailing_activated") else "SL"
                    pos["current_price"] = pos["sl_price"]

            if should_close:
                await self._close_position(pos, close_reason)
            else:
                updated.append(pos)

        self.state.open_positions = updated

    async def _close_position(self, pos, reason):
        exit_price = pos["current_price"]
        direction = pos["direction"]
        entry = pos["entry_price"]

        if direction == 1:
            price_change = (exit_price - entry) / entry
        else:
            price_change = (entry - exit_price) / entry

        trade_value = pos["size_usd"] * pos["leverage"]
        gross_pnl = trade_value * price_change
        fee = trade_value * 0.00055
        net_pnl = gross_pnl - fee

        self.state.demo_balance = self.state.demo_balance + net_pnl

        self.risk_ctrl.record_trade(pos["symbol"], net_pnl, self._candle_idx)
        self.risk_ctrl.update_balance(self.state.demo_balance, self.state.get("demo_balance_initial", 500.0))

        trade = {
            "id": pos["id"],
            "symbol": pos["symbol"],
            "direction": pos["direction"],
            "direction_str": pos["direction_str"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "size_usd": pos["size_usd"],
            "leverage": pos["leverage"],
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(price_change * 100, 2),
            "reason": reason,
            "trailing_activated": pos.get("trailing_activated", False),
            "confidence": pos.get("confidence", 0),
            "regime": pos.get("regime", "normal"),
            "open_time": pos["open_time"],
            "close_time": datetime.utcnow().isoformat(),
            "session_id": pos.get("session_id"),
        }
        self.state.add_trade(trade)

        logger.info(f"CLOSE {pos['direction_str']} {pos['symbol']} | "
                     f"Reason: {reason} | PnL: ${net_pnl:.2f} | "
                     f"Trail: {pos.get('trailing_activated', False)}")

        if self._on_trade_callback:
            await self._on_trade_callback("close", trade)

    async def _close_all_positions(self, reason="SESSION_END"):
        positions = self.state.open_positions
        for pos in positions:
            price = self.prices.get_price(pos["symbol"])
            if price:
                pos["current_price"] = price
            await self._close_position(pos, reason)
        self.state.open_positions = []
