"""Модуль для автоматической торговли на ByBit (demo/testnet)"""
import logging
import hmac
import hashlib
import time
import requests
from typing import Optional, Dict
from urllib.parse import urlencode

import config
from models import Signal


class ByBitTrader:
    """Класс для работы с ByBit API для открытия позиций"""
    
    def __init__(self):
        self.api_key = config.BYBIT_API_KEY
        self.api_secret = config.BYBIT_API_SECRET
        self.base_url = config.BYBIT_FUTURES_BASE
        
        if not self.api_key or not self.api_secret:
            logging.warning("ByBit API ключи не заданы. Торговля будет отключена.")
            self.enabled = False
        else:
            self.enabled = True
    
    def _generate_signature(self, params: dict) -> str:
        """Генерирует подпись для запроса"""
        query_string = urlencode(sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Выполняет запрос к ByBit API"""
        if not self.enabled:
            raise RuntimeError("ByBit API ключи не настроены")
        
        if params is None:
            params = {}
        
        # Добавляем обязательные параметры
        params["api_key"] = self.api_key
        params["timestamp"] = int(time.time() * 1000)
        params["recv_window"] = 5000
        
        # Генерируем подпись
        signature = self._generate_signature(params)
        params["sign"] = signature
        
        url = f"{self.base_url}{endpoint}"
        
        if method.upper() == "GET":
            resp = requests.get(url, params=params, timeout=10)
        else:
            # ByBit v5 API требует параметры в query string для POST запросов
            resp = requests.post(url, params=params, timeout=10)
        
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") != 0:
            error_msg = data.get("retMsg", "Unknown error")
            raise RuntimeError(f"ByBit API error: {error_msg}")
        
        return data
    
    def get_account_info(self) -> dict:
        """Получает информацию об аккаунте"""
        endpoint = "/v5/account/wallet-balance"
        params = {
            "accountType": "UNIFIED",  # Unified account для demo
        }
        return self._make_request("GET", endpoint, params)
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Получает информацию о символе (лот, шаг цены и т.д.)"""
        endpoint = "/v5/market/instruments-info"
        params = {
            "category": "linear",
            "symbol": symbol,
        }
        return self._make_request("GET", endpoint, params)
    
    def calculate_position_size(self, symbol: str, entry_price: float, sl_price: float, risk_percent: float = 1.0) -> float:
        """Рассчитывает размер позиции на основе риска
        
        Args:
            symbol: Торговая пара
            entry_price: Цена входа
            sl_price: Цена стоп-лосса
            risk_percent: Процент баланса для риска (по умолчанию 1%)
        
        Returns:
            Размер позиции в контрактах
        """
        try:
            account_info = self.get_account_info()
            result = account_info.get("result", {})
            balance_list = result.get("list", [])
            
            if not balance_list:
                logging.warning("Не удалось получить баланс аккаунта")
                return 0.0
            
            # Получаем баланс USDT
            total_equity = 0.0
            for account in balance_list:
                coins = account.get("coin", [])
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        total_equity = float(coin.get("walletBalance", 0))
                        break
                if total_equity > 0:
                    break
            
            if total_equity == 0:
                logging.warning(f"❌ Баланс USDT равен 0 для {symbol}")
                return 0.0
            
            logging.info(f"Баланс USDT: {total_equity:.2f}")
            
            # Рассчитываем риск в долларах
            risk_amount = total_equity * (risk_percent / 100.0)
            logging.info(f"Риск на сделку: ${risk_amount:.2f} ({risk_percent}% от баланса ${total_equity:.2f})")
            
            # Рассчитываем риск на контракт
            risk_per_contract = abs(entry_price - sl_price)
            
            if risk_per_contract == 0:
                logging.warning(f"❌ Риск на контракт равен 0 для {symbol} (entry: {entry_price:.6g}, SL: {sl_price:.6g})")
                return 0.0
            
            logging.info(f"Риск на контракт: {risk_per_contract:.6g}")
            
            # Получаем информацию о символе для получения размера лота
            symbol_info = self.get_symbol_info(symbol)
            result = symbol_info.get("result", {})
            instruments = result.get("list", [])
            
            if not instruments:
                logging.warning(f"Не удалось получить информацию о символе {symbol}")
                return 0.0
            
            instrument = instruments[0]
            lot_size_filter = instrument.get("lotSizeFilter", {})
            qty_step = float(lot_size_filter.get("qtyStep", "1")) if lot_size_filter else 1.0
            
            # Рассчитываем количество контрактов
            qty = risk_amount / risk_per_contract
            logging.info(f"Предварительный размер позиции: {qty:.6g} контрактов")
            
            # Округляем до шага лота
            qty = round(qty / qty_step) * qty_step
            logging.info(f"После округления до шага {qty_step}: {qty:.6g}")
            
            # Минимальный размер позиции
            min_qty = float(lot_size_filter.get("minQty", 0)) if lot_size_filter else 0.0
            if qty < min_qty and min_qty > 0:
                logging.info(f"Размер позиции меньше минимума ({qty:.6g} < {min_qty}), устанавливаем минимум")
                qty = min_qty
            
            if qty == 0:
                logging.warning(f"❌ Итоговый размер позиции равен 0 для {symbol} (возможно, риск слишком мал)")
            
            return qty
            
        except Exception as e:
            logging.error(f"Ошибка при расчете размера позиции: {e}", exc_info=True)
            return 0.0
    
    def place_order(
        self,
        signal: Signal,
        qty: Optional[float] = None,
        risk_percent: float = 1.0
    ) -> Optional[dict]:
        """Открывает позицию на основе сигнала
        
        Args:
            signal: Торговый сигнал
            qty: Размер позиции в контрактах (если None, рассчитывается автоматически)
            risk_percent: Процент баланса для риска (используется если qty не задан)
        
        Returns:
            Результат размещения ордера или None при ошибке
        """
        if not self.enabled:
            logging.warning("ByBit торговля отключена (нет API ключей)")
            return None
        
        try:
            # Конвертируем сторону
            side = "Buy" if signal.side == "LONG" else "Sell"
            
            # Рассчитываем размер позиции если не задан
            if qty is None:
                logging.info(f"Рассчитываем размер позиции для {signal.symbol} (риск: {risk_percent}%, entry: {signal.entry:.6g}, SL: {signal.sl:.6g})")
                qty = self.calculate_position_size(
                    signal.symbol,
                    signal.entry,
                    signal.sl,
                    risk_percent
                )
                logging.info(f"Рассчитанный размер позиции для {signal.symbol}: {qty}")
            
            if qty == 0:
                logging.warning(f"❌ Размер позиции для {signal.symbol} равен 0, пропускаем (возможно, недостаточно баланса или проблема с расчетом)")
                return None
            
            # Открываем рыночную позицию
            endpoint = "/v5/order/create"
            params = {
                "category": "linear",
                "symbol": signal.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "positionIdx": 0,  # 0 = односторонняя позиция
            }
            
            order_result = self._make_request("POST", endpoint, params)
            order_id = order_result.get("result", {}).get("orderId")
            
            if not order_id:
                logging.error(f"Не удалось получить orderId для {signal.symbol}")
                return None
            
            logging.info(f"Открыта позиция {signal.side} {signal.symbol}: qty={qty}, orderId={order_id}")
            
            # Устанавливаем стоп-лосс
            sl_result = self._set_stop_loss(signal, order_id)
            
            # Устанавливаем тейк-профиты
            tp_result = self._set_take_profits(signal, order_id)
            
            return {
                "orderId": order_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "qty": qty,
                "entry": signal.entry,
                "sl": signal.sl,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "sl_set": sl_result is not None,
                "tp_set": tp_result is not None,
            }
            
        except Exception as e:
            logging.error(f"Ошибка при открытии позиции для {signal.symbol}: {e}", exc_info=True)
            return None
    
    def _set_stop_loss(self, signal: Signal, order_id: str) -> Optional[dict]:
        """Устанавливает стоп-лосс для позиции"""
        try:
            endpoint = "/v5/position/trading-stop"
            params = {
                "category": "linear",
                "symbol": signal.symbol,
                "stopLoss": str(signal.sl),
                "positionIdx": 0,
            }
            
            result = self._make_request("POST", endpoint, params)
            logging.info(f"Установлен SL для {signal.symbol}: {signal.sl}")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при установке SL для {signal.symbol}: {e}", exc_info=True)
            return None
    
    def _set_take_profits(self, signal: Signal, order_id: str) -> Optional[dict]:
        """Устанавливает тейк-профиты для позиции
        
        Примечание: ByBit поддерживает только один TP через API.
        Устанавливаем TP1, а TP2 можно установить вручную или через частичное закрытие.
        """
        try:
            # Ждем немного, чтобы позиция точно открылась
            time.sleep(0.5)
            
            endpoint = "/v5/position/trading-stop"
            params = {
                "category": "linear",
                "symbol": signal.symbol,
                "takeProfit": str(signal.tp1),  # Устанавливаем TP1
                "positionIdx": 0,
            }
            
            result = self._make_request("POST", endpoint, params)
            logging.info(f"Установлен TP1 для {signal.symbol}: {signal.tp1}")
            
            # Для TP2 можно разместить лимитный ордер на частичное закрытие
            # Но это сложнее, пока устанавливаем только TP1
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при установке TP для {signal.symbol}: {e}", exc_info=True)
            return None


# Глобальный экземпляр трейдера
_trader = None


def get_trader() -> ByBitTrader:
    """Получает глобальный экземпляр трейдера"""
    global _trader
    if _trader is None:
        _trader = ByBitTrader()
    return _trader

