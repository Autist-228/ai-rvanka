import os
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bot_data")
SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")


class UserState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.file_path = os.path.join(STATE_DIR, f"user_{user_id}.json")
        self._data = self._load()
        self._live_prices = {}
        self._last_update_ts = 0

    def _load(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                return json.load(f)
        return self._default_data()

    def _default_data(self):
        return {
            "user_id": self.user_id,
            "demo_balance": 500.0,
            "demo_balance_initial": 500.0,
            "trading_active": False,
            "session_counter": 0,
            "current_session_id": None,
            "session_start_balance": 500.0,
            "session_start_time": None,
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
    def current_session_id(self):
        return self._data.get("current_session_id")

    @property
    def session_counter(self):
        return self._data.get("session_counter", 0)

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
        new_num = self._data.get("session_counter", 0) + 1
        self._data["session_counter"] = new_num
        session_id = f"session_{new_num}"
        self._data["current_session_id"] = session_id
        self._data["trading_active"] = True
        self._data["session_start_balance"] = self.demo_balance
        self._data["session_start_time"] = datetime.utcnow().isoformat()
        self.save()
        logger.info(f"Started session #{new_num} (id={session_id})")

    def stop_session(self):
        self._data["trading_active"] = False
        session_id = self._data.get("current_session_id")
        start_bal = self._data.get("session_start_balance", self.demo_balance)
        session_pnl = self.demo_balance - start_bal

        session_trades = [t for t in self.trade_history if t.get("session_id") == session_id]
        wins = sum(1 for t in session_trades if t.get("pnl", 0) > 0)

        session_data = {
            "session_id": session_id,
            "session_number": self._data.get("session_counter", 0),
            "start_time": self._data.get("session_start_time"),
            "end_time": datetime.utcnow().isoformat(),
            "start_balance": start_bal,
            "end_balance": self.demo_balance,
            "pnl": round(session_pnl, 2),
            "pnl_pct": round(session_pnl / max(start_bal, 1) * 100, 2),
            "total_trades": len(session_trades),
            "wins": wins,
            "win_rate": round(wins / max(len(session_trades), 1) * 100, 1),
            "trades": session_trades,
        }

        self._save_session(session_data)
        self._data["current_session_id"] = None
        self.save()
        logger.info(f"Stopped {session_id}: PnL=${session_pnl:.2f} ({len(session_trades)} trades)")

    def _save_session(self, session_data):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        session_id = session_data.get("session_id", "unknown")
        path = os.path.join(SESSIONS_DIR, f"{self.user_id}_{session_id}.json")
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)
        logger.info(f"Saved session to {path}")

    def get_all_sessions(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        sessions = []
        prefix = f"{self.user_id}_session_"
        for fname in sorted(os.listdir(SESSIONS_DIR)):
            if fname.startswith(prefix) and fname.endswith(".json"):
                try:
                    with open(os.path.join(SESSIONS_DIR, fname), "r") as f:
                        sessions.append(json.load(f))
                except Exception:
                    pass
        return sessions

    def get_session(self, session_number):
        path = os.path.join(SESSIONS_DIR, f"{self.user_id}_session_{session_number}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

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

    def reset_balance(self):
        self._data["demo_balance"] = 500.0
        self._data["demo_balance_initial"] = 500.0
        self._data["session_start_balance"] = 500.0
        self._data["open_positions"] = []
        self._data["stats"] = {
            "total_trades": 0,
            "total_wins": 0,
            "total_pnl": 0.0,
            "daily_pnl": {},
        }
        self.save()
        logger.info(f"Balance reset to $500 for user {self.user_id}")
