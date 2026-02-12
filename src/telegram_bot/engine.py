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

logger = logging.getLogger(__name__)

PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")


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
            xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost{self.model_suffix}.pkl")
            fc_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols{self.model_suffix}.pkl")

            if not os.path.exists(lgb_path) or not os.path.exists(fc_path):
                logger.warning(f"Missing model for {symbol}")
                continue

            self.models[symbol] = {
                "lgb": joblib.load(lgb_path),
                "xgb": joblib.load(xgb_path) if os.path.exists(xgb_path) else None,
            }
            self.feature_cols[symbol] = joblib.load(fc_path)
            logger.info(f"Loaded models for {symbol}")

        self._loaded = True
        logger.info(f"Loaded {len(self.models)} coin models")

    def predict(self, symbol, features_df):
        if symbol not in self.models:
            return None

        model_data = self.models[symbol]
        trained_fc = self.feature_cols[symbol]

        df = features_df.copy()
        missing = [c for c in trained_fc if c not in df.columns]
        for col in missing:
            df[col] = 0
        df[trained_fc] = df[trained_fc].fillna(0)

        X = df[trained_fc].values[-1:].reshape(1, -1)

        lgb_model = model_data["lgb"]
        lgb_pred = lgb_model.predict(X)[0]
        lgb_proba = lgb_model.predict_proba(X)[0]

        label_map = {0: 0, 1: 1, 2: -1}
        lgb_direction = label_map.get(lgb_pred, 0)
        lgb_confidence = lgb_proba[lgb_pred]

        xgb_model = model_data.get("xgb")
        if xgb_model is not None:
            xgb_pred = xgb_model.predict(X)[0]
            xgb_proba = xgb_model.predict_proba(X)[0]
            xgb_direction = label_map.get(xgb_pred, 0)
            xgb_confidence = xgb_proba[xgb_pred]

            if lgb_direction == xgb_direction:
                confidence = (lgb_confidence + xgb_confidence) / 2
                return {"direction": lgb_direction, "confidence": round(confidence, 4), "ensemble": "agree"}

            best_dir, best_conf, best_src = 0, 0.0, "disagree"
            if lgb_direction != 0 and lgb_confidence > best_conf:
                best_dir, best_conf, best_src = lgb_direction, lgb_confidence, "lgb_lead"
            if xgb_direction != 0 and xgb_confidence > best_conf:
                best_dir, best_conf, best_src = xgb_direction, xgb_confidence, "xgb_lead"
            if best_dir != 0 and best_conf >= 0.45:
                return {"direction": best_dir, "confidence": round(best_conf * 0.90, 4), "ensemble": best_src}

            return {"direction": 0, "confidence": 0.0, "ensemble": "disagree"}
        else:
            confidence = lgb_confidence

        return {
            "direction": lgb_direction,
            "confidence": round(confidence, 4),
            "ensemble": "single",
        }


class PriceTracker:
    def __init__(self):
        self._prices = {}
        self._kline_cache = {}

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

    def get_price(self, symbol):
        return self._prices.get(symbol, {}).get("last")

    def get_all_prices(self):
        return dict(self._prices)


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

        settings = self.state.settings
        positions = self.state.open_positions
        max_concurrent = settings.get("max_concurrent_total", 3)

        if len(positions) >= max_concurrent:
            return

        active_symbols = {p["symbol"] for p in positions}
        conf_threshold = settings.get("confidence_threshold", 0.65)
        logger.info(f"Signal check: {len(positions)}/{max_concurrent} pos, threshold={conf_threshold:.2f}")

        for symbol in SYMBOLS:
            if symbol in active_symbols:
                continue
            if len(positions) >= max_concurrent:
                break

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
                    logger.warning(f"{symbol}: missing {PRIMARY_TF} klines, skip")
                    continue

                features_df = self._build_live_features(symbol, klines)
                if features_df is None or len(features_df) < 50:
                    logger.warning(f"{symbol}: features={0 if features_df is None else len(features_df)}, skip")
                    continue

                prediction = self.predictor.predict(symbol, features_df)
                if prediction is None:
                    logger.warning(f"{symbol}: prediction=None, skip")
                    continue

                direction = prediction["direction"]
                confidence = prediction["confidence"]
                dir_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}.get(direction, "?")
                logger.info(f"{symbol}: {dir_str} conf={confidence:.3f} (need>{conf_threshold:.2f})")

                if direction == 0:
                    continue
                if confidence < conf_threshold:
                    continue

                await self._open_position(symbol, prediction, features_df)
                positions = self.state.open_positions

            except Exception as e:
                logger.error(f"Signal check error {symbol}: {e}")

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

    async def _open_position(self, symbol, prediction, features_df):
        settings = self.state.settings
        balance = self.state.demo_balance
        price = self.prices.get_price(symbol)

        if price is None:
            return

        max_pos_pct = settings.get("max_position_pct", 0.05)
        max_pos_usd = settings.get("max_position_usd", 300.0)
        size_usd = min(balance * max_pos_pct, max_pos_usd)

        if size_usd < 1:
            return

        conf = prediction["confidence"]
        conf_threshold = settings.get("confidence_threshold", 0.65)
        max_lev = settings.get("max_leverage", 20)
        min_lev = settings.get("min_leverage", 1)

        conf_ratio = min(max((conf - conf_threshold) / (1.0 - conf_threshold), 0.0), 1.0)
        leverage = int(min_lev + conf_ratio * (max_lev - min_lev))
        leverage = max(min_lev, min(max_lev, leverage))

        coin_tpsl = settings.get("coin_tp_sl", {}).get(symbol, {"tp_pct": 0.04, "sl_pct": 0.02})
        tp_pct = coin_tpsl.get("tp_pct", 0.04)
        sl_pct = coin_tpsl.get("sl_pct", 0.02)

        direction = prediction["direction"]
        if direction == 1:
            tp_price = round(price * (1 + tp_pct), 6)
            sl_price = round(price * (1 - sl_pct), 6)
        else:
            tp_price = round(price * (1 - tp_pct), 6)
            sl_price = round(price * (1 + sl_pct), 6)

        position = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "direction": direction,
            "direction_str": "LONG" if direction == 1 else "SHORT",
            "entry_price": price,
            "current_price": price,
            "size_usd": round(size_usd, 2),
            "leverage": leverage,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "confidence": conf,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "open_time": datetime.utcnow().isoformat(),
            "session_start": self.state.get("session_start_time"),
        }

        self.state.add_position(position)

        fee = size_usd * leverage * 0.00055
        self.state.demo_balance = self.state.demo_balance - fee

        logger.info(f"Opened {position['direction_str']} {symbol} @ {price} | "
                     f"Size: ${size_usd:.2f} | Lev: {leverage}x | Conf: {conf:.3f}")

        if self._on_trade_callback:
            await self._on_trade_callback("open", position)

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
                if price >= pos["tp_price"]:
                    should_close = True
                    close_reason = "TP"
                elif price <= pos["sl_price"]:
                    should_close = True
                    close_reason = "SL"
            else:
                if price <= pos["tp_price"]:
                    should_close = True
                    close_reason = "TP"
                elif price >= pos["sl_price"]:
                    should_close = True
                    close_reason = "SL"

            if should_close:
                await self._close_position(pos, close_reason)
            else:
                updated.append(pos)

        self.state.open_positions = updated

    async def _close_position(self, pos, reason):
        fee = pos["size_usd"] * pos["leverage"] * 0.00055
        net_pnl = pos["pnl"] - fee

        self.state.demo_balance = self.state.demo_balance + net_pnl

        trade = {
            "id": pos["id"],
            "symbol": pos["symbol"],
            "direction_str": pos["direction_str"],
            "entry_price": pos["entry_price"],
            "exit_price": pos["current_price"],
            "size_usd": pos["size_usd"],
            "leverage": pos["leverage"],
            "pnl": round(net_pnl, 2),
            "pnl_pct": pos["pnl_pct"],
            "reason": reason,
            "open_time": pos["open_time"],
            "close_time": datetime.utcnow().isoformat(),
            "session_start": pos.get("session_start"),
        }
        self.state.add_trade(trade)

        logger.info(f"Closed {pos['direction_str']} {pos['symbol']} | "
                     f"Reason: {reason} | PnL: ${net_pnl:.2f}")

        if self._on_trade_callback:
            await self._on_trade_callback("close", trade)

    async def _close_all_positions(self, reason="SESSION_END"):
        positions = self.state.open_positions
        for pos in positions:
            price = self.prices.get_price(pos["symbol"])
            if price:
                pos["current_price"] = price
                direction = pos["direction"]
                entry = pos["entry_price"]
                if direction == 1:
                    price_change = (price - entry) / entry
                else:
                    price_change = (entry - price) / entry
                trade_value = pos["size_usd"] * pos["leverage"]
                pos["pnl"] = round(trade_value * price_change, 2)
                pos["pnl_pct"] = round(price_change * 100, 2)

            await self._close_position(pos, reason)

        self.state.open_positions = []
