import os
import time
import math
import logging
import asyncio
from io import BytesIO
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Bot, InputFile
from telegram.constants import ParseMode

# =======================
#  НАСТРОЙКИ И ЛОГИ
# =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "600"))
TOP_N = int(os.getenv("TOP_N", "8"))
MIN_QUOTE_VOLUME_USDT = float(os.getenv("MIN_QUOTE_VOLUME_USDT", "1000000"))

TIMEFRAME_MAIN = os.getenv("TIMEFRAME_MAIN", "5m")
TIMEFRAME_TREND = os.getenv("TIMEFRAME_TREND", "1h")

RSI_LONG_MIN = float(os.getenv("RSI_LONG_MIN", "50"))
RSI_LONG_MAX = float(os.getenv("RSI_LONG_MAX", "72"))
RSI_SHORT_MIN = float(os.getenv("RSI_SHORT_MIN", "28"))
RSI_SHORT_MAX = float(os.getenv("RSI_SHORT_MAX", "50"))

VOL_SPIKE_MULTIPLIER = float(os.getenv("VOL_SPIKE_MULTIPLIER", "1.5"))

ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
ATR_TP1_MULTIPLIER = float(os.getenv("ATR_TP1_MULTIPLIER", "2.0"))
ATR_TP2_MULTIPLIER = float(os.getenv("ATR_TP2_MULTIPLIER", "3.0"))

BTC_TREND_FILTER = int(os.getenv("BTC_TREND_FILTER", "1"))

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в .env")

bot = Bot(token=TELEGRAM_BOT_TOKEN)


# =======================
#  ХЕЛПЕРЫ ПО ИНДИКАТОРАМ
# =======================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    avg_gain = avg_gain.shift(1) * (period - 1) / period + gain / period
    avg_loss = avg_loss.shift(1) * (period - 1) / period + loss / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=period).mean()
    return atr_series


# =======================
#  BINANCE API
# =======================

def get_24h_tickers() -> List[Dict]:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    filtered = [
        x for x in data
        if x.get("symbol", "").endswith("USDT")
    ]
    return filtered


def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise RuntimeError(f"No kline data for {symbol} {interval}")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


# =======================
#  СТРУКТУРЫ ДАННЫХ
# =======================

@dataclass
class Signal:
    symbol: str
    side: str
    reason: str
    timeframe: str
    trend_tf: str
    last_price: float
    rsi: float
    ema_fast: float
    ema_slow: float
    atr: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    volume_24h: float
    change_24h: float
    tag: str


# =======================
#  ЛОГИКА ТОП-МУВЕРОВ
# =======================

def select_top_movers(tickers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    df = pd.DataFrame(tickers)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    df["quoteVolume"] = df["quoteVolume"].astype(float)

    df = df[df["quoteVolume"] >= MIN_QUOTE_VOLUME_USDT]

    gainers = df.sort_values("priceChangePercent", ascending=False).head(TOP_N)
    losers = df.sort_values("priceChangePercent", ascending=True).head(TOP_N)

    return gainers.to_dict("records"), losers.to_dict("records")


def detect_market_trend_btc() -> str:
    try:
        df = get_klines("BTCUSDT", TIMEFRAME_TREND, limit=200)
    except Exception as e:
        logging.warning("Не удалось получить BTCUSDT для тренда: %s", e)
        return "UNKNOWN"

    close = df["close"]
    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)
    rsi_val = rsi(close, 14)

    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])

    if math.isnan(last_ema_fast) or math.isnan(last_ema_slow) or math.isnan(last_rsi):
        return "UNKNOWN"

    if last_ema_fast > last_ema_slow and last_rsi > 55:
        return "UP"
    elif last_ema_fast < last_ema_slow and last_rsi < 45:
        return "DOWN"
    else:
        return "SIDE"


def build_signal(symbol: str, side: str, ticker_row: Dict, market_trend: str) -> Optional[Signal]:
    try:
        df_main = get_klines(symbol, TIMEFRAME_MAIN, limit=200)
        _ = get_klines(symbol, TIMEFRAME_TREND, limit=200)
    except Exception as e:
        logging.warning("Klines error for %s: %s", symbol, e)
        return None

    close = df_main["close"]
    vol = df_main["volume"]

    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)
    rsi_series = rsi(close, 14)
    atr_series = atr(df_main, 14)

    last_close = float(close.iloc[-1])
    last_rsi = float(rsi_series.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_atr = float(atr_series.iloc[-1])

    if any(math.isnan(x) for x in [last_rsi, last_ema_fast, last_ema_slow, last_atr]):
        return None

    avg_vol = float(vol.iloc[-50:].mean())
    last_vol = float(vol.iloc[-1])
    vol_spike = last_vol > VOL_SPIKE_MULTIPLIER * avg_vol if avg_vol > 0 else False

    trend_ok = False
    rsi_ok = False

    if side == "LONG":
        trend_ok = last_ema_fast > last_ema_slow
        rsi_ok = RSI_LONG_MIN <= last_rsi <= RSI_LONG_MAX
    else:
        trend_ok = last_ema_fast < last_ema_slow
        rsi_ok = RSI_SHORT_MIN <= last_rsi <= RSI_SHORT_MAX

    if not (trend_ok and rsi_ok and vol_spike):
        return None

    if BTC_TREND_FILTER and market_trend in ("UP", "DOWN"):
        if side == "LONG" and market_trend == "DOWN":
            return None
        if side == "SHORT" and market_trend == "UP":
            return None

    if side == "LONG":
        entry = last_close
        sl = entry - ATR_SL_MULTIPLIER * last_atr
        tp1 = entry + ATR_TP1_MULTIPLIER * last_atr
        tp2 = entry + ATR_TP2_MULTIPLIER * last_atr
    else:
        entry = last_close
        sl = entry + ATR_SL_MULTIPLIER * last_atr
        tp1 = entry - ATR_TP1_MULTIPLIER * last_atr
        tp2 = entry - ATR_TP2_MULTIPLIER * last_atr

    price_change = float(ticker_row["priceChangePercent"])
    high_price = float(ticker_row["highPrice"])
    low_price = float(ticker_row["lowPrice"])
    last_price = float(ticker_row["lastPrice"])

    tag_parts = []
    if side == "LONG" and price_change > 5:
        tag_parts.append("Rally")
    if side == "SHORT" and price_change < -5:
        tag_parts.append("Dump")
    if last_price > 0.98 * high_price:
        tag_parts.append("Near 24h High")
    if last_price < 1.02 * low_price:
        tag_parts.append("Near 24h Low")

    tag = ", ".join(tag_parts) if tag_parts else "Normal"

    return Signal(
        symbol=symbol,
        side=side,
        reason=f"Trend & RSI & Volume spike ({TIMEFRAME_MAIN})",
        timeframe=TIMEFRAME_MAIN,
        trend_tf=TIMEFRAME_TREND,
        last_price=last_close,
        rsi=last_rsi,
        ema_fast=last_ema_fast,
        ema_slow=last_ema_slow,
        atr=last_atr,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        volume_24h=float(ticker_row["quoteVolume"]),
        change_24h=price_change,
        tag=tag,
    )


# =======================
#  TELEGRAM ФОРМАТ
# =======================

def format_signals_message(
    market_trend: str,
    gain_signals: List[Signal],
    loss_signals: List[Signal],
) -> str:
    """Форматирует сообщение для отправки в .txt файл (без HTML тегов)"""
    if not gain_signals and not loss_signals:
        return "🤖 Binance Top Movers\n\nСигналов по фильтрам пока нет. Рынок спит."

    lines = []
    lines.append("🤖 Binance Futures Top Movers Signals")
    lines.append("")
    lines.append(f"🌍 Market (BTC {TIMEFRAME_TREND}): {market_trend}")
    lines.append("")

    def fmt_sig(sig: Signal) -> str:
        rr1 = abs((sig.tp1 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0
        rr2 = abs((sig.tp2 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0

        return (
            f"{sig.symbol} {sig.side}\n"
            f"\n"
            f"🏷 Tag: {sig.tag}\n"
            f"\n"
            f"📊 24h Change: {sig.change_24h:+.2f}%\n"
            f"💰 Price: {sig.last_price:.6g}\n"
            f"\n"
            f"📈 Indicators:\n"
            f"  RSI14: {sig.rsi:.1f}\n"
            f"  EMA20: {sig.ema_fast:.6g}\n"
            f"  EMA50: {sig.ema_slow:.6g}\n"
            f"  ATR14: {sig.atr:.6g}\n"
            f"\n"
            f"🎯 Levels:\n"
            f"  Entry: {sig.entry:.6g}\n"
            f"  SL: {sig.sl:.6g}\n"
            f"  TP1: {sig.tp1:.6g} (RR≈{rr1:.1f})\n"
            f"  TP2: {sig.tp2:.6g} (RR≈{rr2:.1f})\n"
        )

    if gain_signals:
        lines.append("📈 LONG candidates (Top Gainers)")
        lines.append("")
        for sig in gain_signals:
            lines.append(fmt_sig(sig))
            lines.append("")
    if loss_signals:
        if gain_signals:
            lines.append("")
        lines.append("📉 SHORT candidates (Top Losers)")
        lines.append("")
        for sig in loss_signals:
            lines.append(fmt_sig(sig))
            lines.append("")

    lines.append("")
    lines.append("⚠️ Это не финансовый совет. Торгуй головой, а не печени.")

    return "\n".join(lines)


def format_signals_message_console(
    market_trend: str,
    gain_signals: List[Signal],
    loss_signals: List[Signal],
) -> str:
    """Форматирует сообщение для вывода в консоль (без HTML-тегов)"""
    if not gain_signals and not loss_signals:
        return "🤖 Binance Top Movers\n\nСигналов по фильтрам пока нет. Рынок спит."

    lines = []
    lines.append("🤖 Binance Futures Top Movers Signals")
    lines.append("")
    lines.append(f"🌍 Market (BTC {TIMEFRAME_TREND}): {market_trend}")
    lines.append("")

    def fmt_sig(sig: Signal) -> str:
        rr1 = abs((sig.tp1 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0
        rr2 = abs((sig.tp2 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0

        return (
            f"{sig.symbol} {sig.side}\n"
            f"• Tag: {sig.tag}\n"
            f"• 24h Chg: {sig.change_24h:+.2f}%\n"
            f"• Price: {sig.last_price:.6g}\n"
            f"• RSI14: {sig.rsi:.1f} | EMA20: {sig.ema_fast:.6g} | EMA50: {sig.ema_slow:.6g}\n"
            f"• ATR14: {sig.atr:.6g}\n"
            f"• Entry: {sig.entry:.6g}\n"
            f"• SL: {sig.sl:.6g}\n"
            f"• TP1: {sig.tp1:.6g} (RR≈{rr1:.1f})\n"
            f"• TP2: {sig.tp2:.6g} (RR≈{rr2:.1f})\n"
        )

    if gain_signals:
        lines.append("📈 LONG candidates (Top Gainers)")
        for sig in gain_signals:
            lines.append(fmt_sig(sig))
    if loss_signals:
        lines.append("")
        lines.append("📉 SHORT candidates (Top Losers)")
        for sig in loss_signals:
            lines.append(fmt_sig(sig))

    lines.append("")
    lines.append("⚠️ Это не финансовый совет. Торгуй головой, а не печени.")

    return "\n".join(lines)


async def send_telegram_file_async(content: str, filename: str):
    """Асинхронная отправка файла в Telegram"""
    # Создаем файл в памяти
    file_obj = BytesIO(content.encode('utf-8'))
    file_obj.name = filename
    
    # Используем InputFile для правильной отправки файла
    input_file = InputFile(file_obj, filename=filename)
    
    await bot.send_document(
        chat_id=TELEGRAM_CHAT_ID,
        document=input_file,
        caption=filename.replace('.txt', '').replace('_', ' ').title()
    )


async def send_telegram_files_async(files: List[Tuple[str, str]]):
    """Асинхронная отправка нескольких файлов в Telegram
    files: список кортежей (content, filename)
    """
    for content, filename in files:
        await send_telegram_file_async(content, filename)


def send_telegram_file(content: str, filename: str):
    """Синхронная обертка для отправки файла в Telegram"""
    send_telegram_files([(content, filename)])


def send_telegram_files(files: List[Tuple[str, str]]):
    """Синхронная обертка для отправки нескольких файлов в Telegram в одном event loop
    files: список кортежей (content, filename)
    """
    if not files:
        return
    
    try:
        # Создаем один event loop для всех файлов
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(send_telegram_files_async(files))
        finally:
            loop.close()
    except Exception as e:
        logging.error("Ошибка при отправке файлов в Telegram: %s", e)
        raise


# =======================
#  ОСНОВНОЙ ЦИКЛ
# =======================

def format_logs_message(logs: List[str], gainers: List[Dict], losers: List[Dict]) -> str:
    """Форматирует логи итерации для отправки в .txt файл"""
    lines = []
    lines.append("Логи итерации")
    lines.append("")
    
    # Основная информация
    for log in logs:
        # Убираем символы типа ✓ и другие спецсимволы
        log_line = log.replace("✓", "").strip()
        if log_line:
            lines.append(log_line)
            lines.append("")
    
    # Добавляем список топ-пар
    lines.append("")
    lines.append("=" * 50)
    lines.append("Топ Gainers (растущие монеты):")
    lines.append("")
    if gainers:
        for i, row in enumerate(gainers, 1):
            symbol = row.get("symbol", "")
            change = float(row.get("priceChangePercent", 0))
            volume = float(row.get("quoteVolume", 0))
            lines.append(f"{i}. {symbol}")
            lines.append(f"   Изменение: {change:+.2f}%")
            lines.append(f"   Объем 24h: ${volume:,.0f}")
            lines.append("")
    else:
        lines.append("Нет данных")
        lines.append("")
    
    lines.append("=" * 50)
    lines.append("Топ Losers (падающие монеты):")
    lines.append("")
    if losers:
        for i, row in enumerate(losers, 1):
            symbol = row.get("symbol", "")
            change = float(row.get("priceChangePercent", 0))
            volume = float(row.get("quoteVolume", 0))
            lines.append(f"{i}. {symbol}")
            lines.append(f"   Изменение: {change:+.2f}%")
            lines.append(f"   Объем 24h: ${volume:,.0f}")
            lines.append("")
    else:
        lines.append("Нет данных")
        lines.append("")
    
    return "\n".join(lines)


def run_once():
    # Собираем логи итерации
    iteration_logs: List[str] = []
    
    logging.info("Старт сканирования Top Movers...")
    iteration_logs.append("Старт сканирования Top Movers...")
    
    tickers = get_24h_tickers()
    gainers, losers = select_top_movers(tickers)
    logging.info("Отобрано gainers=%d, losers=%d", len(gainers), len(losers))
    iteration_logs.append(f"Отобрано gainers={len(gainers)}, losers={len(losers)}")

    market_trend = detect_market_trend_btc()
    logging.info("BTC market trend: %s", market_trend)
    iteration_logs.append(f"BTC market trend: {market_trend}")

    gain_signals: List[Signal] = []
    loss_signals: List[Signal] = []

    checked_symbols = []
    for row in gainers:
        sym = row["symbol"]
        logging.info("Проверяем LONG %s", sym)
        checked_symbols.append(f"LONG {sym}")
        sig = build_signal(sym, "LONG", row, market_trend)
        if sig:
            gain_signals.append(sig)
            iteration_logs.append(f"LONG {sym} - сигнал найден")

    for row in losers:
        sym = row["symbol"]
        logging.info("Проверяем SHORT %s", sym)
        checked_symbols.append(f"SHORT {sym}")
        sig = build_signal(sym, "SHORT", row, market_trend)
        if sig:
            loss_signals.append(sig)
            iteration_logs.append(f"SHORT {sym} - сигнал найден")
    
    iteration_logs.append(f"Проверено символов: {len(checked_symbols)}")
    iteration_logs.append(f"Найдено сигналов: LONG={len(gain_signals)}, SHORT={len(loss_signals)}")

    msg = format_signals_message(market_trend, gain_signals, loss_signals)
    msg_console = format_signals_message_console(market_trend, gain_signals, loss_signals)
    
    # Вывод в консоль
    print("\n" + "=" * 80)
    print(msg_console)
    print("=" * 80 + "\n")
    
    try:
        # Отправляем файлы в одном event loop
        logs_msg = format_logs_message(iteration_logs, gainers, losers)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_to_send = []
        
        # Отправляем файл с сигналами только если есть хотя бы один сигнал
        if gain_signals or loss_signals:
            files_to_send.append((msg, f"signals_{timestamp}.txt"))
            logging.info("Найдены сигналы, будет отправлен файл signals.")
        else:
            logging.info("Сигналов не найдено, файл signals не отправляется.")
        
        # Логи отправляем всегда
        files_to_send.append((logs_msg, f"logs_{timestamp}.txt"))
        
        if files_to_send:
            send_telegram_files(files_to_send)
            logging.info("Отправлены файлы в Telegram.")
    except Exception as e:
        logging.warning("Не удалось отправить файлы в Telegram: %s", e)


def main():
    logging.info("Запускаем Binance Top Movers bot (FULL mode).")
    while True:
        try:
            run_once()
        except Exception as e:
            logging.exception("Ошибка в run_once: %s", e)
            try:
                error_msg = f"⚠️ Бот поймал ошибку:\n{str(e)}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                send_telegram_file(error_msg, f"error_{timestamp}.txt")
            except Exception as telegram_error:
                logging.warning("Не удалось отправить ошибку в Telegram: %s", telegram_error)
        logging.info("Спим %d секунд...", SCAN_INTERVAL_SECONDS)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
