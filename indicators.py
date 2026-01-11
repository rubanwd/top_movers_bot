"""Модуль технических индикаторов"""
import pandas as pd
import numpy as np
from typing import Tuple


def ema(series: pd.Series, period: int) -> pd.Series:
    """Вычисляет экспоненциальную скользящую среднюю (EMA)"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Вычисляет индекс относительной силы (RSI)"""
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
    """Вычисляет средний истинный диапазон (ATR)"""
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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Вычисляет MACD: возвращает (macd_line, signal_line, histogram)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Вычисляет ADX (Average Directional Index) для определения силы тренда"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    # Smoothing
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)
    
    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_series = dx.rolling(window=period).mean()
    
    return adx_series


