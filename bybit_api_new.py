"""
Модуль для работы с ByBit Futures API v5 с правильной аутентификацией
"""

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
        self.key = api_key or getattr(config, "BYBIT_API_KEY", "") or ""
        self.secret = api_secret or getattr(config, "BYBIT_API_SECRET", "") or ""

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

    # -------- Market --------

    def get_tickers(self, category="linear", symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/market/tickers", params)

    def get_kline(self, category="linear", symbol="BTCUSDT", interval="60", limit=200) -> Dict[str, Any]:
        params = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        return self._get("/v5/market/kline", params)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        params = {"category": "linear", "symbol": symbol}
        return self._get("/v5/market/instruments-info", params)

    # -------- Account / Trade --------

    def get_account_info(self) -> Dict[str, Any]:
        params = {"accountType": "UNIFIED"}
        return self._get("/v5/account/wallet-balance", params, auth=True)

    def get_open_orders(self, category="linear", symbol: Optional[str] = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/order/realtime", params, auth=True)

    def get_positions(self, category="linear", symbol: Optional[str] = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get("/v5/position/list", params, auth=True)

    def set_position_mode(self, category="linear", mode="one_way", coin="USDT", symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Устанавливает режим позиции:
        - one_way -> mode=0
        - hedge  -> mode=1

        ВАЖНО: Bybit требует coin или symbol (иначе coin and symbol cannot be both empty)
        """
        m = 0 if mode == "one_way" else 1
        payload: Dict[str, Any] = {"category": category, "mode": m, "coin": coin}
        if symbol:
            payload["symbol"] = symbol
        return self._post("/v5/position/switch-mode", payload, auth=True)

    def set_leverage(self, category="linear", symbol="BTCUSDT", buy_leverage=10, sell_leverage=10) -> Dict[str, Any]:
        payload = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        return self._post("/v5/position/set-leverage", payload, auth=True)

    def place_order(self, **kwargs) -> Dict[str, Any]:
        return self._post("/v5/order/create", kwargs, auth=True)

    def cancel_order(self, category="linear", symbol="", orderId: Optional[str] = None, orderLinkId: Optional[str] = None):
        payload = {"category": category, "symbol": symbol}
        if orderId:
            payload["orderId"] = orderId
        if orderLinkId:
            payload["orderLinkId"] = orderLinkId
        return self._post("/v5/order/cancel", payload, auth=True)

    def set_sl_tp(
        self,
        category="linear",
        symbol="",
        positionIdx: int = 0,
        takeProfit: Optional[str] = None,
        stopLoss: Optional[str] = None,
        tpTriggerBy="LastPrice",
        slTriggerBy="LastPrice",
        reduceOnly=True,
    ):
        payload = {"category": category, "symbol": symbol, "positionIdx": str(positionIdx)}
        if takeProfit is not None:
            payload["takeProfit"] = str(takeProfit)
        if stopLoss is not None:
            payload["stopLoss"] = str(stopLoss)
        payload["tpTriggerBy"] = tpTriggerBy
        payload["slTriggerBy"] = slTriggerBy
        payload["reduceOnly"] = "true" if reduceOnly else "false"
        return self._post("/v5/position/trading-stop", payload, auth=True)

    # -------- HTTP helpers --------

    def _get(self, path: str, params: Dict[str, Any], auth: bool = False) -> Dict[str, Any]:
        url = self.base + path
        headers = self._auth_headers(params, is_get=True) if auth else None
        r = requests.get(url, params=params, headers=headers, timeout=30)

        if r.status_code == 401:
            error_data = r.json() if r.content else {}
            error_msg = error_data.get("retMsg", "API key is invalid")
            ret_code = error_data.get("retCode", "?")
            logging.error("❌ ByBit API: Невалидный API ключ (401). Проверьте BYBIT_API_KEY/BYBIT_API_SECRET")
            logging.error(f"   retCode: {ret_code}, retMsg: {error_msg}")
            logging.error(f"   base_url: {self.base}")
            raise RuntimeError(f"ByBit API authentication failed: {error_msg}")

        r.raise_for_status()
        data = r.json()

        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown error")
            ret_code = data.get("retCode", "?")
            raise RuntimeError(f"ByBit API error (retCode={ret_code}): {error_msg}")

        return data

    def _post(self, path: str, body: Dict[str, Any], auth: bool = False) -> Dict[str, Any]:
        """
        Важно: для Bybit v5 подпись POST генерируется из JSON строки тела.
        """
        url = self.base + path

        if auth:
            json_body = json.dumps(body)  # сохраняем порядок ключей как есть
            headers = self._auth_headers(body, is_get=False, json_body_str=json_body)
            r = requests.post(url, data=json_body, headers=headers, timeout=30)
        else:
            headers = {"Content-Type": "application/json"}
            r = requests.post(url, json=body, headers=headers, timeout=30)

        if r.status_code == 401:
            error_data = r.json() if r.content else {}
            error_msg = error_data.get("retMsg", "API key is invalid")
            logging.error("❌ ByBit API: Невалидный API ключ (401). Проверьте BYBIT_API_KEY/BYBIT_API_SECRET")
            logging.error(f"   base_url: {self.base}")
            raise RuntimeError(f"ByBit API authentication failed: {error_msg}")

        r.raise_for_status()
        data = r.json()

        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown error")
            ret_code = data.get("retCode", "?")
            raise RuntimeError(f"ByBit API error (retCode={ret_code}): {error_msg}")

        return data

    def _auth_headers(self, payload: Dict[str, Any], is_get: bool = True, json_body_str: str = None) -> Dict[str, str]:
        if not self.key or not self.secret:
            raise RuntimeError("API ключи не настроены")

        ts = str(int(time.time() * 1000))
        recv_window = "20000"

        if payload:
            if is_get:
                sorted_items = sorted([(str(k), str(v)) for k, v in payload.items()])
                body_str = urlencode(sorted_items)
            else:
                body_str = json_body_str if json_body_str is not None else json.dumps(payload, separators=(",", ":"))
        else:
            body_str = ""

        to_sign = ts + self.key + recv_window + body_str
        sign = hmac.new(self.secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()

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
    global _api_instance
    if _api_instance is None:
        _api_instance = BybitAPI()
    return _api_instance


# -------- Backward compatible helpers --------

def get_24h_tickers() -> List[Dict]:
    api = get_api()
    tick = api.get_tickers(category="linear")
    result = tick.get("result", {})
    tickers = result.get("list", []) or []

    filtered = [
        {
            "symbol": x.get("symbol", ""),
            "lastPrice": float(x.get("lastPrice", 0)),
            "priceChangePercent": float(x.get("price24hPcnt", 0)) * 100,
            "quoteVolume": float(x.get("turnover24h", 0)),
            "highPrice": float(x.get("highPrice24h", 0)),
            "lowPrice": float(x.get("lowPrice24h", 0)),
        }
        for x in tickers
        if x.get("symbol", "").endswith("USDT")
    ]
    return filtered


def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
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

    lst.reverse()

    num_cols = len(lst[0]) if lst else 0
    if num_cols == 7:
        cols = ["startTime", "open", "high", "low", "close", "volume", "turnover"]
    elif num_cols == 8:
        cols = ["startTime", "open", "high", "low", "close", "volume", "turnover", "ignore"]
    else:
        raise RuntimeError(f"Неожиданный формат данных от ByBit: {num_cols} колонок для {symbol}")

    df = pd.DataFrame(lst, columns=cols)

    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["close_time"] = df["open_time"]
    df["quote_asset_volume"] = df["turnover"]
    df["number_of_trades"] = 0
    df["taker_buy_base"] = 0
    df["taker_buy_quote"] = 0

    return df
