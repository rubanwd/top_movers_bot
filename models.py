"""Модели данных для бота"""
from dataclasses import dataclass


@dataclass
class Signal:
    """Структура данных для торгового сигнала"""
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
    score: float = 0.0  # Оценка качества сигнала (0-100)


