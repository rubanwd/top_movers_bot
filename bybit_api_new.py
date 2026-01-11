"""Модуль для работы с ByBit Futures API v5 с правильной аутентификацией"""
import time
import hmac
import hashlib
import requests
import pandas as pd
import logging
import json
from urllib.parse import urlencode
from typing import Dict, Any, Optional, List
import config


class BybitAPI:
    """Класс для работы с ByBit API v5"""
    
    def __init__(self, base_url: str = None, api_key: str = None, api_secret: str = None):
        self.base = (base_url or config.BYBIT_FUTURES_BASE).rstrip("/")
        self.key = api_key or config.BYBIT_API_KEY or ""
        self.secret = api_secret or config.BYBIT_API_SECRET or ""
        
        # Логируем настройки при инициализации
        if self.key and self.secret:
            logging.info(f"ByBit API инициализирован: base_url={self.base}")
            logging.info(f"  API Key: {self.key[:10]}... (первые 10 символов)")
            if "demo" in self.base.lower():
                logging.info("  ⚠️ Используется DEMO API (демо-торговля)")
            elif "testnet" in self.base.lower():
                logging.info("  ⚠️ Используется TESTNET (демо-торговля)")
            else:
                logging.info("  ⚠️ Используется MAINNET (реальная торговля!)")
        else:
            logging.warning("ByBit API ключи не заданы")
    
    def get_tickers(self, category="linear", symbol: Optional[str] = None) -> Dict[str, Any]:
        """Получает список тикеров"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/market/tickers", params)
    
    def get_kline(self, category="linear", symbol="BTCUSDT", interval="60", limit=200) -> Dict[str, Any]:
        """Получает свечи (klines)"""
        params = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        return self._get("/v5/market/kline", params)
    
    def get_account_info(self) -> Dict[str, Any]:
        """Получает информацию об аккаунте"""
        params = {"accountType": "UNIFIED"}
        return self._get("/v5/account/wallet-balance", params, auth=True)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Получает информацию о символе"""
        params = {"category": "linear", "symbol": symbol}
        return self._get("/v5/market/instruments-info", params)
    
    def set_position_mode(self, category="linear", mode="one_way") -> Dict[str, Any]:
        """Устанавливает режим позиции (one_way или hedge)"""
        m = 0 if mode == "one_way" else 1
        payload = {"category": category, "mode": m}
        return self._post("/v5/position/switch-mode", payload, auth=True)
    
    def set_leverage(self, category="linear", symbol="BTCUSDT", buy_leverage=10, sell_leverage=10) -> Dict[str, Any]:
        """Устанавливает кредитное плечо"""
        payload = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage)
        }
        return self._post("/v5/position/set-leverage", payload, auth=True)
    
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Размещает ордер"""
        return self._post("/v5/order/create", kwargs, auth=True)
    
    def cancel_order(self, category="linear", symbol="", orderId: Optional[str] = None, orderLinkId: Optional[str] = None):
        """Отменяет ордер"""
        payload = {"category": category, "symbol": symbol}
        if orderId:
            payload["orderId"] = orderId
        if orderLinkId:
            payload["orderLinkId"] = orderLinkId
        return self._post("/v5/order/cancel", payload, auth=True)
    
    def get_open_orders(self, category="linear", symbol: Optional[str] = None):
        """Получает открытые ордера"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/order/realtime", params, auth=True)
    
    def get_positions(self, category="linear", symbol: Optional[str] = None):
        """Получает открытые позиции"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/position/list", params, auth=True)
    
    def set_sl_tp(self, category="linear", symbol="", positionIdx: int = 0,
                  takeProfit: Optional[str] = None, stopLoss: Optional[str] = None,
                  tpTriggerBy="LastPrice", slTriggerBy="LastPrice", reduceOnly=True):
        """Устанавливает стоп-лосс и тейк-профит"""
        payload = {"category": category, "symbol": symbol, "positionIdx": str(positionIdx)}
        if takeProfit is not None:
            payload["takeProfit"] = str(takeProfit)
        if stopLoss is not None:
            payload["stopLoss"] = str(stopLoss)
        payload["tpTriggerBy"] = tpTriggerBy
        payload["slTriggerBy"] = slTriggerBy
        payload["reduceOnly"] = "true" if reduceOnly else "false"
        return self._post("/v5/position/trading-stop", payload, auth=True)
    
    def _get(self, path: str, params: Dict[str, Any], auth: bool = False) -> Dict[str, Any]:
        """Выполняет GET запрос"""
        url = self.base + path
        headers = self._auth_headers(params, is_get=True) if auth else None
        r = requests.get(url, params=params, headers=headers, timeout=30)
        
        # Проверяем статус ответа
        if r.status_code == 401:
            error_data = r.json() if r.content else {}
            error_msg = error_data.get("retMsg", "API key is invalid")
            ret_code = error_data.get("retCode", "?")
            
            logging.error(f"❌ ByBit API: Невалидный API ключ (401). Проверьте BYBIT_API_KEY и BYBIT_API_SECRET в .env")
            logging.error(f"   retCode: {ret_code}, retMsg: {error_msg}")
            logging.error(f"   Убедитесь, что используете правильные ключи для {self.base}")
            logging.error(f"   URL: {url}")
            logging.error(f"   Параметры: {params}")
            logging.error(f"   API Key длина: {len(self.key)} символов")
            logging.error(f"   API Secret длина: {len(self.secret)} символов")
            logging.error(f"   Проверьте:")
            logging.error(f"     1. Ключи скопированы полностью (без пробелов)")
            logging.error(f"     2. Ключи для правильной среды (testnet/mainnet)")
            logging.error(f"     3. У ключа есть права Read и Trade")
            logging.error(f"     4. Ключи не истекли и не были удалены")
            
            raise RuntimeError(f"ByBit API authentication failed: {error_msg}")
        
        r.raise_for_status()
        data = r.json()
        
        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown error")
            ret_code = data.get("retCode", "?")
            raise RuntimeError(f"ByBit API error (retCode={ret_code}): {error_msg}")
        
        return data
    
    def _post(self, path: str, body: Dict[str, Any], auth: bool = False) -> Dict[str, Any]:
        """Выполняет POST запрос
        
        Важно: Тело отправляется как JSON, и подпись генерируется из той же JSON строки.
        """
        url = self.base + path
        
        # Для POST запросов с аутентификацией: генерируем JSON строку для подписи
        # и используем её же для отправки тела
        if auth:
            # Создаем JSON строку БЕЗ сортировки ключей (сохраняем исходный порядок)
            # Используем стандартный json.dumps (с пробелами, как в ошибке Bybit)
            json_body = json.dumps(body)
            headers = self._auth_headers(body, is_get=False, json_body_str=json_body)
            # Отправляем уже сериализованную JSON строку
            r = requests.post(url, data=json_body, headers=headers, timeout=30)
        else:
            headers = {"Content-Type": "application/json"}
            r = requests.post(url, json=body, headers=headers, timeout=30)
        
        # Проверяем статус ответа
        if r.status_code == 401:
            error_data = r.json() if r.content else {}
            error_msg = error_data.get("retMsg", "API key is invalid")
            logging.error(f"❌ ByBit API: Невалидный API ключ (401). Проверьте BYBIT_API_KEY и BYBIT_API_SECRET в .env")
            logging.error(f"   Убедитесь, что используете правильные ключи для {self.base}")
            raise RuntimeError(f"ByBit API authentication failed: {error_msg}")
        
        r.raise_for_status()
        data = r.json()
        
        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown error")
            ret_code = data.get("retCode", "?")
            raise RuntimeError(f"ByBit API error (retCode={ret_code}): {error_msg}")
        
        return data
    
    def _auth_headers(self, payload: Dict[str, Any], is_get: bool = True, json_body_str: str = None) -> Dict[str, str]:
        """Генерирует заголовки для аутентификации
        
        Важно: Для Bybit API v5:
        - GET запросы: подпись генерируется из urlencode(sorted(payload.items()))
        - POST запросы: подпись генерируется из JSON строки (json_body_str, если предоставлена)
        - POST тело отправляется как JSON, и подпись тоже из JSON формата
        
        Все значения в payload должны быть строками для правильной генерации подписи.
        """
        if not self.key or not self.secret:
            raise RuntimeError("API ключи не настроены")
        
        ts = str(int(time.time() * 1000))
        recv_window = "20000"  # Увеличено до 20 секунд для компенсации рассинхронизации времени
        
        # Для Bybit API v5: разные форматы для GET и POST
        if payload:
            if is_get:
                # GET: используем urlencode
                sorted_items = sorted([(str(k), str(v)) for k, v in payload.items()])
                body_str = urlencode(sorted_items)
            else:
                # POST: используем предоставленную JSON строку или создаем новую
                if json_body_str is not None:
                    body_str = json_body_str
                else:
                    # Fallback: создаем JSON строку (сортировка ключей)
                    sorted_payload = {k: payload[k] for k in sorted(payload.keys())}
                    body_str = json.dumps(sorted_payload, separators=(',', ':'))
        else:
            body_str = ""
        
        to_sign = ts + self.key + recv_window + body_str
        sign = hmac.new(self.secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
        
        # Логирование для отладки подписи (временно включено для диагностики)
        logging.info(f"[AUTH DEBUG] payload: {payload}")
        logging.info(f"[AUTH DEBUG] body_str: {body_str}")
        logging.info(f"[AUTH DEBUG] to_sign (first 200 chars): {to_sign[:200]}")
        logging.info(f"[AUTH DEBUG] sign: {sign}")
        
        return {
            "X-BAPI-API-KEY": self.key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }


# Глобальный экземпляр API
_api_instance = None


def get_api() -> BybitAPI:
    """Получает глобальный экземпляр API"""
    global _api_instance
    if _api_instance is None:
        _api_instance = BybitAPI()
    return _api_instance


# Функции для обратной совместимости
def get_24h_tickers() -> List[Dict]:
    """Получает список всех тикеров с 24ч статистикой (обратная совместимость)"""
    api = get_api()
    tick = api.get_tickers(category="linear")
    result = tick.get("result", {})
    tickers = result.get("list", []) or []
    
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
    """Получает свечи (klines) для символа (обратная совместимость)"""
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
    
    api = get_api()
    resp = api.get_kline(category="linear", symbol=symbol, interval=bybit_interval, limit=limit)
    
    lst = (resp.get("result", {}) or {}).get("list", []) or []
    if not lst:
        return pd.DataFrame()
    
    # ByBit возвращает данные в обратном порядке (новые первыми), нужно перевернуть
    lst.reverse()
    
    # ByBit v5 формат: [startTime, open, high, low, close, volume, turnover] - 7 колонок
    num_cols = len(lst[0]) if lst else 0
    if num_cols == 7:
        cols = ["startTime", "open", "high", "low", "close", "volume", "turnover"]
    elif num_cols == 8:
        cols = ["startTime", "open", "high", "low", "close", "volume", "turnover", "ignore"]
    else:
        raise RuntimeError(f"Неожиданный формат данных от ByBit: {num_cols} колонок для {symbol}")
    
    df = pd.DataFrame(lst, columns=cols)
    
    # Конвертируем в числовые типы
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Переименовываем и добавляем колонки для совместимости
    df["open_time"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["close_time"] = df["open_time"]
    df["quote_asset_volume"] = df["turnover"]
    df["number_of_trades"] = 0
    df["taker_buy_base"] = 0
    df["taker_buy_quote"] = 0
    
    return df

