"""Модуль для работы с ByBit Futures API v5"""
import requests
import pandas as pd
import time
import hmac
import hashlib
from typing import List, Dict, Optional
import config


def _generate_signature(api_secret: str, params: dict) -> str:
    """Генерирует подпись для запроса"""
    param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(
        api_secret.encode("utf-8"),
        param_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def get_24h_tickers() -> List[Dict]:
    """Получает список всех тикеров с 24ч статистикой"""
    url = f"{config.BYBIT_FUTURES_BASE}/v5/market/tickers"
    params = {
        "category": "linear"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
    
    tickers = data.get("result", {}).get("list", [])
    # Фильтруем только USDT пары
    filtered = [t for t in tickers if t.get("symbol", "").endswith("USDT")]
    
    # Преобразуем формат для совместимости с остальным кодом
    result = []
    for ticker in filtered:
        # Bybit возвращает price24hPcnt как число (например, 0.05 для 5%)
        # Нужно преобразовать в проценты (5.0 для 5%)
        price_change_pct = float(ticker.get("price24hPcnt", "0") or "0") * 100
        
        result.append({
            "symbol": ticker.get("symbol", ""),
            "lastPrice": ticker.get("lastPrice", "0"),
            "priceChangePercent": str(price_change_pct),  # В процентах как строка для совместимости
            "highPrice": ticker.get("highPrice24h", "0"),
            "lowPrice": ticker.get("lowPrice24h", "0"),
            "quoteVolume": ticker.get("turnover24h", "0"),
            "volume": ticker.get("volume24h", "0"),
        })
    
    return result


def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Получает свечи (klines) для символа"""
    # Конвертируем интервал из Binance формата в Bybit формат
    # Bybit использует: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
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
    
    # Если интервал уже в формате Bybit, используем как есть
    bybit_interval = interval_map.get(interval, interval)
    
    # Если интервал не найден, пробуем использовать как есть (может быть уже в формате Bybit)
    if bybit_interval not in ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]:
        # Пробуем преобразовать числовые значения
        try:
            if interval.endswith("m"):
                bybit_interval = interval[:-1]
            elif interval.endswith("h"):
                bybit_interval = str(int(interval[:-1]) * 60)
            else:
                bybit_interval = interval
        except:
            bybit_interval = interval
    
    url = f"{config.BYBIT_FUTURES_BASE}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
    
    raw = data.get("result", {}).get("list", [])
    if not raw:
        raise RuntimeError(f"No kline data for {symbol} {interval}")
    
    # Bybit возвращает данные в обратном порядке (от старых к новым)
    # Преобразуем в формат, совместимый с pandas
    df_data = []
    for item in raw:
        df_data.append({
            "open_time": int(item[0]),  # startTime
            "open": float(item[1]),     # open
            "high": float(item[2]),     # high
            "low": float(item[3]),      # low
            "close": float(item[4]),    # close
            "volume": float(item[5]),   # volume
            "close_time": int(item[0]), # endTime (используем startTime как close_time для совместимости)
            "quote_asset_volume": float(item[6]) if len(item) > 6 else 0.0,  # turnover
        })
    
    df = pd.DataFrame(df_data)
    
    # Сортируем по времени (от старых к новым)
    df = df.sort_values("open_time").reset_index(drop=True)
    
    return df


class BybitAPI:
    """Класс для работы с Bybit API v5"""
    
    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _request(self, method: str, endpoint: str, params: Optional[dict] = None, signed: bool = False) -> dict:
        """Выполняет HTTP запрос к API"""
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params["api_key"] = self.api_key
            params["timestamp"] = str(int(time.time() * 1000))
            params["recv_window"] = "5000"
            params["sign"] = _generate_signature(self.api_secret, params)
        
        if method.upper() == "GET":
            resp = requests.get(url, params=params, timeout=10)
        else:
            resp = requests.post(url, json=params, timeout=10)
        
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'Unknown error')} (code: {data.get('retCode')})")
        
        return data
    
    def get_account_info(self) -> dict:
        """Получает информацию об аккаунте"""
        return self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}, signed=True)
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Получает информацию о символе"""
        return self._request("GET", "/v5/market/instruments-info", {
            "category": "linear",
            "symbol": symbol
        })
    
    def get_positions(self, category: str = "linear", symbol: Optional[str] = None) -> dict:
        """Получает список открытых позиций"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/v5/position/list", params, signed=True)
    
    def set_position_mode(self, category: str, mode: str, coin: str) -> dict:
        """Устанавливает режим позиций (one-way или hedge)"""
        params = {
            "category": category,
            "mode": mode,
            "coin": coin
        }
        return self._request("POST", "/v5/position/switch-mode", params, signed=True)
    
    def set_leverage(self, category: str, symbol: str, buy_leverage: int, sell_leverage: int) -> dict:
        """Устанавливает плечо для символа"""
        params = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage)
        }
        return self._request("POST", "/v5/position/set-leverage", params, signed=True)
    
    def place_order(self, category: str, symbol: str, side: str, orderType: str, qty: str, positionIdx: int, **kwargs) -> dict:
        """Размещает ордер"""
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": orderType,
            "qty": qty,
            "positionIdx": positionIdx,
            **kwargs
        }
        return self._request("POST", "/v5/order/create", params, signed=True)
    
    def set_sl_tp(self, category: str, symbol: str, positionIdx: int, stopLoss: Optional[str] = None, takeProfit: Optional[str] = None) -> dict:
        """Устанавливает стоп-лосс и/или тейк-профит"""
        params = {
            "category": category,
            "symbol": symbol,
            "positionIdx": positionIdx
        }
        if stopLoss:
            params["stopLoss"] = stopLoss
        if takeProfit:
            params["takeProfit"] = takeProfit
        return self._request("POST", "/v5/position/trading-stop", params, signed=True)


def get_api() -> Optional[BybitAPI]:
    """Фабричная функция для создания экземпляра BybitAPI"""
    api_key = getattr(config, "BYBIT_API_KEY", None)
    api_secret = getattr(config, "BYBIT_API_SECRET", None)
    base_url = getattr(config, "BYBIT_FUTURES_BASE", "https://api-demo.bybit.com")
    
    if not api_key or not api_secret:
        return None
    
    return BybitAPI(base_url=base_url, api_key=api_key, api_secret=api_secret)
