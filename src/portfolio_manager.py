import numpy as np
import pandas as pd


class RiskController:
    def __init__(self, config):
        self.max_portfolio_drawdown_pct = config.get("max_portfolio_drawdown_pct", 0.20)
        self.drawdown_leverage_cut = config.get("drawdown_leverage_cut", 0.5)
        self.consecutive_sl_pause = config.get("consecutive_sl_pause", 3)
        self.pause_candles = config.get("pause_candles", 4)
        self.coin_loss_streak_reduce = config.get("coin_loss_streak_reduce", 3)
        self.coin_loss_streak_factor = config.get("coin_loss_streak_factor", 0.5)
        self.compounding_threshold = config.get("compounding_threshold", 0.10)
        self.compounding_factor = config.get("compounding_factor", 1.2)
        self.trailing_activation_pct = config.get("trailing_activation_pct", 0.02)
        self.trailing_step_pct = config.get("trailing_step_pct", 0.01)

        self.peak_balance = 0
        self.current_drawdown = 0
        self.consecutive_losses = {}
        self.global_consecutive_losses = 0
        self.pause_until = {}
        self.coin_trade_history = {}

    def update_balance(self, balance, initial_balance):
        if balance > self.peak_balance:
            self.peak_balance = balance
        self.current_drawdown = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0

    def record_trade(self, symbol, pnl, candle_idx):
        if symbol not in self.coin_trade_history:
            self.coin_trade_history[symbol] = []
        self.coin_trade_history[symbol].append(pnl)

        if symbol not in self.consecutive_losses:
            self.consecutive_losses[symbol] = 0

        if pnl <= 0:
            self.consecutive_losses[symbol] += 1
            self.global_consecutive_losses += 1
        else:
            self.consecutive_losses[symbol] = 0
            self.global_consecutive_losses = 0

        if self.consecutive_losses[symbol] >= self.consecutive_sl_pause:
            self.pause_until[symbol] = candle_idx + self.pause_candles
            self.consecutive_losses[symbol] = 0

    def is_paused(self, symbol, candle_idx):
        if symbol in self.pause_until:
            if candle_idx < self.pause_until[symbol]:
                return True
            else:
                del self.pause_until[symbol]
        return False

    def get_leverage_multiplier(self, balance, initial_balance):
        if self.current_drawdown > self.max_portfolio_drawdown_pct:
            return self.drawdown_leverage_cut

        profit_pct = (balance - initial_balance) / initial_balance
        if profit_pct > self.compounding_threshold:
            return min(self.compounding_factor, 1.0 + profit_pct * 0.5)

        return 1.0

    def get_coin_allocation_factor(self, symbol):
        losses = self.consecutive_losses.get(symbol, 0)
        if losses >= self.coin_loss_streak_reduce:
            return self.coin_loss_streak_factor
        return 1.0


class MetaAllocator:
    def __init__(self, config):
        self.max_single_coin_pct = config.get("max_single_coin_pct", 0.40)
        self.min_single_coin_pct = config.get("min_single_coin_pct", 0.05)
        self.confidence_power = config.get("confidence_power", 2.0)
        self.correlation_penalty = config.get("correlation_penalty", 0.7)
        self.regime_cash_pct = config.get("regime_cash_pct", 0.50)
        self.no_signal_cash = config.get("no_signal_cash", True)

        self.correlation_pairs = {
            ("BTCUSDT", "ETHUSDT"): 0.85,
            ("BTCUSDT", "BNBUSDT"): 0.75,
            ("BTCUSDT", "SOLUSDT"): 0.70,
            ("ETHUSDT", "SOLUSDT"): 0.72,
            ("ETHUSDT", "BNBUSDT"): 0.65,
            ("XRPUSDT", "DOGEUSDT"): 0.60,
        }

    def allocate(self, signals, risk_controller):
        active_signals = {}
        for symbol, sig in signals.items():
            if sig["direction"] == 0:
                continue
            if sig["confidence"] < sig.get("threshold", 0.55):
                continue
            if risk_controller.is_paused(symbol, sig.get("candle_idx", 0)):
                continue
            active_signals[symbol] = sig

        if not active_signals:
            return {}

        raw_weights = {}
        for symbol, sig in active_signals.items():
            conf_score = (sig["confidence"] - sig.get("threshold", 0.55)) ** self.confidence_power
            coin_factor = risk_controller.get_coin_allocation_factor(symbol)
            raw_weights[symbol] = conf_score * coin_factor

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            return {}

        allocations = {}
        for symbol, w in raw_weights.items():
            pct = w / total_weight
            pct = min(pct, self.max_single_coin_pct)
            pct = max(pct, self.min_single_coin_pct)
            allocations[symbol] = pct

        for s1 in list(allocations.keys()):
            for s2 in list(allocations.keys()):
                if s1 >= s2:
                    continue
                pair = (s1, s2)
                reverse_pair = (s2, s1)
                corr = self.correlation_pairs.get(pair, self.correlation_pairs.get(reverse_pair, 0))
                if corr > 0.7:
                    combined = allocations[s1] + allocations[s2]
                    if combined > self.max_single_coin_pct * 1.5:
                        if raw_weights.get(s1, 0) >= raw_weights.get(s2, 0):
                            allocations[s2] *= self.correlation_penalty
                        else:
                            allocations[s1] *= self.correlation_penalty

        has_crash = any(sig.get("regime") == "crash" for sig in active_signals.values())
        if has_crash:
            for symbol in allocations:
                allocations[symbol] *= (1 - self.regime_cash_pct)

        total = sum(allocations.values())
        if total > 1.0:
            for symbol in allocations:
                allocations[symbol] /= total

        return allocations


PORTFOLIO_CONFIG = {
    "initial_balance": 500.0,

    "risk_controller": {
        "max_portfolio_drawdown_pct": 0.15,
        "drawdown_leverage_cut": 0.5,
        "consecutive_sl_pause": 3,
        "pause_candles": 4,
        "coin_loss_streak_reduce": 3,
        "coin_loss_streak_factor": 0.5,
        "daily_loss_pause_candles": 96,
        "compounding_threshold": 0.20,
        "compounding_factor": 1.1,
        "trailing_activation_pct": 0.02,
        "trailing_step_pct": 0.01,
    },

    "allocator": {
        "max_single_coin_pct": 0.05,
        "min_single_coin_pct": 0.02,
        "max_position_usd": 300.0,
        "confidence_power": 2.0,
        "correlation_penalty": 0.7,
        "regime_cash_pct": 0.50,
        "no_signal_cash": True,
    },

    "trading": {
        "max_leverage": 20,
        "min_leverage": 1,
        "base_tp_pct": 0.04,
        "base_sl_pct": 0.02,
        "fee_pct": 0.0012,
        "max_concurrent_per_coin": 1,
        "max_concurrent_total": 6,
        "confidence_threshold": 0.55,
    },

    "coin_tp_sl": {
        "BTCUSDT":  {"tp_pct": 0.025, "sl_pct": 0.012, "atr_tp_mult": 2.0, "atr_sl_mult": 1.0},
        "ETHUSDT":  {"tp_pct": 0.035, "sl_pct": 0.018, "atr_tp_mult": 2.2, "atr_sl_mult": 1.1},
        "SOLUSDT":  {"tp_pct": 0.045, "sl_pct": 0.022, "atr_tp_mult": 2.5, "atr_sl_mult": 1.2},
        "XRPUSDT":  {"tp_pct": 0.040, "sl_pct": 0.020, "atr_tp_mult": 2.3, "atr_sl_mult": 1.1},
        "BNBUSDT":  {"tp_pct": 0.030, "sl_pct": 0.015, "atr_tp_mult": 2.0, "atr_sl_mult": 1.0},
        "DOGEUSDT": {"tp_pct": 0.060, "sl_pct": 0.030, "atr_tp_mult": 3.0, "atr_sl_mult": 1.5},
    },
}
