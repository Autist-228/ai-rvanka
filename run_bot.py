import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.telegram_bot.bot import run_bot

if __name__ == "__main__":
    run_bot()
