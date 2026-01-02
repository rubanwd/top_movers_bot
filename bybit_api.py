"""Модуль для работы с ByBit Futures API (demo/testnet)"""
import requests
import pandas as pd
from typing import List, Dict
import config


def get_24h_tickers() -> List[Dict]:
    """Получает список всех тикеров с 24ч статистикой"""
    url = f"{config.BYBIT_FUTURES_BASE}/v5/market/tickers"
    params = {
        "category": "linear",  # USDT perpetual futures
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get("retCode") != 0:
        raise RuntimeError(f"ByBit API error: {data.get('retMsg', 'Unknown error')}")
    
    tickers = data.get("result", {}).get("list", [])
    # Фильтруем только USDT пары
    filtered = [
        {
            "symbol": x.get("symbol", ""),
            "lastPrice": float(x.get("lastPrice", 0)),
            "priceChangePercent": float(x.get("price24hPcnt", 0)) * 100,  # ByBit возвращает в долях
            "quoteVolume": float(x.get("turnover24h", 0)),  # Объем в USDT
            "highPrice": float(x.get("highPrice24h", 0)),
            "lowPrice": float(x.get("lowPrice24h", 0)),
        }
        for x in tickers
        if x.get("symbol", "").endswith("USDT")
    ]
    return filtered


def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Получает свечи (klines) для символа
    
    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Таймфрейм (5m, 15m, 1h и т.д.)
        limit: Количество свечей
    
    Returns:
        DataFrame с колонками: open_time, open, high, low, close, volume, ...
    """
    # Конвертируем интервал в формат ByBit API
    interval_map = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    
    bybit_interval = interval_map.get(interval, interval)
    
    url = f"{config.BYBIT_FUTURES_BASE}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    
    if raw.get("retCode") != 0:
        raise RuntimeError(f"ByBit API error: {raw.get('retMsg', 'Unknown error')}")
    
    klines = raw.get("result", {}).get("list", [])
    if not klines:
        raise RuntimeError(f"No kline data for {symbol} {interval}")
    
    # ByBit возвращает данные в обратном порядке (новые первыми), нужно перевернуть
    klines.reverse()
    
    # ByBit формат: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(klines, columns=[
        "startTime", "open", "high", "low", "close", "volume", "turnover", "ignore"
    ])
    
    # Конвертируем в числовые типы
    df["open_time"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["quote_asset_volume"] = pd.to_numeric(df["turnover"], errors="coerce")
    
    # Добавляем фиктивные колонки для совместимости с существующим кодом
    df["close_time"] = df["open_time"]
    df["number_of_trades"] = 0
    df["taker_buy_base"] = 0
    df["taker_buy_quote"] = 0
    
    return df

