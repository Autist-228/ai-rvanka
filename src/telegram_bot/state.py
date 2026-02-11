import os
import json
import time
from datetime import datetime

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bot_data")

DEFAULT_SETTINGS = {
    "confidence_threshold": 0.65,
    "max_leverage": 20,
    "min_leverage": 1,
    "max_position_pct": 0.05,
    "max_position_usd": 300.0,
    "max_concurrent_total": 3,
    "max_drawdown_pct": 0.15,
    "consecutive_sl_pause": 3,
    "coin_tp_sl": {
        "BTCUSDT":  {"tp_pct": 0.025, "sl_pct": 0.012},
        "ETHUSDT":  {"tp_pct": 0.035, "sl_pct": 0.018},
        "SOLUSDT":  {"tp_pct": 0.045, "sl_pct": 0.022},
        "XRPUSDT":  {"tp_pct": 0.040, "sl_pct": 0.020},
        "BNBUSDT":  {"tp_pct": 0.030, "sl_pct": 0.015},
        "DOGEUSDT": {"tp_pct": 0.060, "sl_pct": 0.030},
    },
}


class UserState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.file_path = os.path.join(STATE_DIR, f"user_{user_id}.json")
        self._data = self._load()

    def _load(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                return json.load(f)
        return self._default_data()

    def _default_data(self):
        return {
            "user_id": self.user_id,
            "active_account": "demo",
            "demo_balance": 500.0,
            "demo_balance_initial": 500.0,
            "trading_active": False,
            "session_start_balance": 500.0,
            "session_start_time": None,
            "last_session": None,
            "settings": dict(DEFAULT_SETTINGS),
            "open_positions": [],
            "trade_history": [],
            "stats": {
                "total_trades": 0,
                "total_wins": 0,
                "total_pnl": 0.0,
                "daily_pnl": {},
            },
            "main_message_id": None,
            "main_chat_id": None,
            "bybit_api_key": None,
            "bybit_api_secret": None,
            "created_at": datetime.utcnow().isoformat(),
        }

    def save(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

    @property
    def active_account(self):
        return self._data.get("active_account", "demo")

    @active_account.setter
    def active_account(self, value):
        self._data["active_account"] = value
        self.save()

    @property
    def demo_balance(self):
        return self._data.get("demo_balance", 500.0)

    @demo_balance.setter
    def demo_balance(self, value):
        self._data["demo_balance"] = round(value, 2)
        self.save()

    @property
    def trading_active(self):
        return self._data.get("trading_active", False)

    @trading_active.setter
    def trading_active(self, value):
        self._data["trading_active"] = value
        self.save()

    @property
    def settings(self):
        s = self._data.get("settings", {})
        for k, v in DEFAULT_SETTINGS.items():
            if k not in s:
                s[k] = v
        return s

    @property
    def open_positions(self):
        return self._data.get("open_positions", [])

    @open_positions.setter
    def open_positions(self, value):
        self._data["open_positions"] = value
        self.save()

    @property
    def trade_history(self):
        return self._data.get("trade_history", [])

    def add_trade(self, trade):
        if "trade_history" not in self._data:
            self._data["trade_history"] = []
        self._data["trade_history"].append(trade)

        stats = self._data.get("stats", {"total_trades": 0, "total_wins": 0, "total_pnl": 0.0, "daily_pnl": {}})
        stats["total_trades"] += 1
        if trade.get("pnl", 0) > 0:
            stats["total_wins"] += 1
        stats["total_pnl"] += trade.get("pnl", 0)

        day = trade.get("close_time", datetime.utcnow().isoformat())[:10]
        if day not in stats.get("daily_pnl", {}):
            stats["daily_pnl"][day] = 0.0
        stats["daily_pnl"][day] += trade.get("pnl", 0)
        self._data["stats"] = stats
        self.save()

    def add_position(self, position):
        if "open_positions" not in self._data:
            self._data["open_positions"] = []
        self._data["open_positions"].append(position)
        self.save()

    def remove_position(self, position_id):
        self._data["open_positions"] = [
            p for p in self._data.get("open_positions", []) if p.get("id") != position_id
        ]
        self.save()

    def start_session(self):
        self._data["trading_active"] = True
        self._data["session_start_balance"] = self.demo_balance
        self._data["session_start_time"] = datetime.utcnow().isoformat()
        self.save()

    def stop_session(self):
        self._data["trading_active"] = False
        session_pnl = self.demo_balance - self._data.get("session_start_balance", self.demo_balance)
        self._data["last_session"] = {
            "start_time": self._data.get("session_start_time"),
            "end_time": datetime.utcnow().isoformat(),
            "start_balance": self._data.get("session_start_balance", self.demo_balance),
            "end_balance": self.demo_balance,
            "pnl": round(session_pnl, 2),
            "pnl_pct": round(session_pnl / max(self._data.get("session_start_balance", 500), 1) * 100, 2),
            "trades": len([t for t in self.trade_history if t.get("session_start") == self._data.get("session_start_time")]),
        }
        self.save()

    def update_setting(self, key, value):
        if "settings" not in self._data:
            self._data["settings"] = dict(DEFAULT_SETTINGS)
        self._data["settings"][key] = value
        self.save()

    def get_pnl_period(self, days):
        now = datetime.utcnow()
        total = 0.0
        daily = self._data.get("stats", {}).get("daily_pnl", {})
        for day_str, pnl in daily.items():
            try:
                day_dt = datetime.fromisoformat(day_str)
                if (now - day_dt).days <= days:
                    total += pnl
            except (ValueError, TypeError):
                pass
        return round(total, 2)

    def get_win_rate(self):
        stats = self._data.get("stats", {})
        total = stats.get("total_trades", 0)
        wins = stats.get("total_wins", 0)
        if total == 0:
            return 0.0
        return round(wins / total * 100, 1)
