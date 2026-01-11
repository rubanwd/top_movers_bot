"""Модуль для работы с ByBit Futures API (demo/testnet) - использует новый API класс"""
from bybit_api_new import get_24h_tickers, get_klines, BybitAPI, get_api

# Экспортируем для обратной совместимости
__all__ = ['get_24h_tickers', 'get_klines', 'BybitAPI', 'get_api']
