"""Модуль для работы с Binance Futures API"""
import requests
import pandas as pd
from typing import List, Dict
import config


def get_24h_tickers() -> List[Dict]:
    """Получает список всех тикеров с 24ч статистикой"""
    url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    filtered = [
        x for x in data
        if x.get("symbol", "").endswith("USDT")
    ]
    return filtered


def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Получает свечи (klines) для символа"""
    url = f"{config.BINANCE_FUTURES_BASE}/fapi/v1/klines"
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


