"""Модуль конфигурации - загрузка настроек из переменных окружения"""
import os
import logging
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Загрузка переменных окружения
load_dotenv()

# Выбор биржи: "bybit" или "binance" (по умолчанию bybit)
EXCHANGE = os.getenv("EXCHANGE", "bybit").lower()

# Binance API
BINANCE_FUTURES_BASE = os.getenv("BINANCE_FUTURES_BASE", "https://fapi.binance.com")

# ByBit API (demo/testnet/mainnet)
# Поддерживает: https://api-demo.bybit.com (demo), https://api-testnet.bybit.com (testnet), https://api.bybit.com (mainnet)
BYBIT_FUTURES_BASE = os.getenv("BYBIT_FUTURES_BASE", "https://api-demo.bybit.com")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_ENABLE_TRADING = int(os.getenv("BYBIT_ENABLE_TRADING", "1"))  # 1 = включено, 0 = выключено
BYBIT_RISK_PERCENT = float(os.getenv("BYBIT_RISK_PERCENT", "1.0"))  # Процент баланса для риска на сделку
# Тестовый режим для отладки открытия позиций (открывает тестовую позицию на BTCUSDT)
TEST_MODE = int(os.getenv("TEST_MODE", "0"))  # 1 = включено (открывает тестовую позицию), 0 = выключено

# Валидация выбора биржи
if EXCHANGE not in ["bybit", "binance"]:
    logging.warning(f"Неизвестная биржа '{EXCHANGE}', используем 'bybit' по умолчанию")
    EXCHANGE = "bybit"

logging.info(f"Используется биржа: {EXCHANGE.upper()}")

# Telegram настройки
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHAT_ID_2 = os.getenv("TELEGRAM_CHAT_ID_2")  # Дополнительный канал для сигналов

# Основные настройки
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "600"))

# Загружаем TOP_N с отладкой
top_n_raw = os.getenv("TOP_N", "8")
logging.info(f"TOP_N из .env (сырое значение): '{top_n_raw}' (тип: {type(top_n_raw).__name__})")
TOP_N = int(top_n_raw.strip()) if top_n_raw else 8
logging.info(f"TOP_N после обработки: {TOP_N}")

MIN_QUOTE_VOLUME_USDT = float(os.getenv("MIN_QUOTE_VOLUME_USDT", "5000000"))  # Увеличено до 5M для более ликвидных пар

# Логируем загруженные значения для отладки
logging.info(f"Загружены настройки: TOP_N={TOP_N}, SCAN_INTERVAL_SECONDS={SCAN_INTERVAL_SECONDS}, MIN_QUOTE_VOLUME_USDT={MIN_QUOTE_VOLUME_USDT}")

# Таймфреймы
TIMEFRAME_MAIN = os.getenv("TIMEFRAME_MAIN", "5m")
TIMEFRAME_TREND = os.getenv("TIMEFRAME_TREND", "1h")

# RSI параметры (ужесточены для более точных сигналов)
RSI_LONG_MIN = float(os.getenv("RSI_LONG_MIN", "50"))  # Ужесточено: было 45
RSI_LONG_MAX = float(os.getenv("RSI_LONG_MAX", "60"))  # Ужесточено: было 65
RSI_SHORT_MIN = float(os.getenv("RSI_SHORT_MIN", "40"))  # Ужесточено: было 35
RSI_SHORT_MAX = float(os.getenv("RSI_SHORT_MAX", "50"))  # Ужесточено: было 55

# Объем (увеличен множитель для очень строгого отбора)
VOL_SPIKE_MULTIPLIER = float(os.getenv("VOL_SPIKE_MULTIPLIER", "2.0"))  # Увеличено до 2.0 для очень строгого отбора

# ATR множители для SL/TP
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
ATR_TP1_MULTIPLIER = float(os.getenv("ATR_TP1_MULTIPLIER", "2.0"))
ATR_TP2_MULTIPLIER = float(os.getenv("ATR_TP2_MULTIPLIER", "3.0"))

# Фильтр тренда BTC (включен для более точных сигналов)
BTC_TREND_FILTER = int(os.getenv("BTC_TREND_FILTER", "1"))  # Включено для более точных сигналов

# Параметры для раннего обнаружения движения
MAX_24H_CHANGE = float(os.getenv("MAX_24H_CHANGE", "20.0"))  # Уменьшено до 20% для избежания перекупленных/перепроданных
USE_MAX_24H_FILTER = int(os.getenv("USE_MAX_24H_FILTER", "1"))
RECENT_CANDLES_LOOKBACK = int(os.getenv("RECENT_CANDLES_LOOKBACK", "2"))
MIN_RECENT_CHANGE_PCT = float(os.getenv("MIN_RECENT_CHANGE_PCT", "0.5"))  # Увеличено до 0.5 для очень значимых движений
# Включаем строгие проверки для более точных сигналов
RECENT_MOVE_CHECK = int(os.getenv("RECENT_MOVE_CHECK", "1"))  # Включено для более точных сигналов
RSI_ENTRY_CHECK = int(os.getenv("RSI_ENTRY_CHECK", "1"))  # Включено для более точных сигналов
EMA_CROSS_RECENT = int(os.getenv("EMA_CROSS_RECENT", "1"))  # Включено для более точных сигналов
VOL_RECENT_CHECK = int(os.getenv("VOL_RECENT_CHECK", "1"))  # Включено для более точных сигналов

# Параметры для улучшенной стратегии
MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "3"))  # Уменьшено до 3 для очень строгого отбора
# Включаем MACD и ADX для очень строгого отбора
USE_MACD = int(os.getenv("USE_MACD", "1"))  # Включено для более точных сигналов
USE_ADX = int(os.getenv("USE_ADX", "1"))  # Включено для более точных сигналов
MIN_ADX = float(os.getenv("MIN_ADX", "30.0"))  # Увеличено до 30 для очень сильных трендов

# Валидация обязательных параметров
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в .env")

