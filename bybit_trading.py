"""Модуль для автоматической торговли на ByBit (demo/testnet)"""
import logging
from typing import Optional, Dict

import config
from models import Signal
from bybit_api_new import BybitAPI


class ByBitTrader:
    """Класс для работы с ByBit API для открытия позиций"""
    
    def __init__(self):
        self.api_key = config.BYBIT_API_KEY
        self.api_secret = config.BYBIT_API_SECRET
        self.base_url = config.BYBIT_FUTURES_BASE
        
        if not self.api_key or not self.api_secret:
            logging.warning("ByBit API ключи не заданы. Торговля будет отключена.")
            self.enabled = False
            self.api = None
        else:
            self.enabled = True
            self.api = BybitAPI(base_url=self.base_url, api_key=self.api_key, api_secret=self.api_secret)
            
            # Устанавливаем режим позиции в one-way (односторонний)
            try:
                self.api.set_position_mode(category="linear", mode="one_way")
                logging.info("✅ Режим позиции установлен: one-way (односторонний)")
            except Exception as e:
                logging.warning(f"⚠️ Не удалось установить режим позиции (возможно, уже установлен): {e}")
    
    def get_account_info(self) -> dict:
        """Получает информацию об аккаунте"""
        if not self.enabled or not self.api:
            raise RuntimeError("ByBit API ключи не настроены")
        return self.api.get_account_info()
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Получает информацию о символе (лот, шаг цены и т.д.)"""
        if not self.enabled or not self.api:
            raise RuntimeError("ByBit API ключи не настроены")
        return self.api.get_symbol_info(symbol)
    
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
            
        except RuntimeError as e:
            # Если ошибка аутентификации, логируем и возвращаем 0
            if "authentication" in str(e).lower() or "401" in str(e) or "invalid" in str(e).lower():
                logging.error(f"❌ Ошибка аутентификации ByBit API. Торговля отключена. Проверьте API ключи в .env")
                logging.error(f"   Используется base_url: {self.base_url}")
                logging.error(f"   Убедитесь, что API ключи правильные для testnet/mainnet")
                return 0.0
            raise
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
            order_result = self.api.place_order(
                category="linear",
                symbol=signal.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                positionIdx="0",  # 0 = односторонняя позиция (строка для правильной подписи)
            )
            
            result_data = order_result.get("result", {})
            order_id = result_data.get("orderId")
            
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
        if not self.enabled or not self.api:
            return None
        
        try:
            result = self.api.set_sl_tp(
                category="linear",
                symbol=signal.symbol,
                positionIdx=0,
                stopLoss=str(signal.sl),
            )
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
        if not self.enabled or not self.api:
            return None
        
        try:
            # Ждем немного, чтобы позиция точно открылась
            import time
            time.sleep(0.5)
            
            result = self.api.set_sl_tp(
                category="linear",
                symbol=signal.symbol,
                positionIdx=0,
                takeProfit=str(signal.tp1),  # Устанавливаем TP1
            )
            logging.info(f"Установлен TP1 для {signal.symbol}: {signal.tp1}")
            
            # Для TP2 можно разместить лимитный ордер на частичное закрытие
            # Но это сложнее, пока устанавливаем только TP1
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при установке TP для {signal.symbol}: {e}", exc_info=True)
            return None


    def test_order_placement(self, symbol: str = "BTCUSDT", side: str = "LONG", risk_percent: float = 1.0) -> Optional[dict]:
        """Тестовая функция для открытия позиции без сигнала (для отладки)
        
        Args:
            symbol: Торговая пара (по умолчанию BTCUSDT)
            side: Направление "LONG" или "SHORT" (по умолчанию LONG)
            risk_percent: Процент баланса для риска (по умолчанию 1%)
        
        Returns:
            Результат размещения ордера или None при ошибке
        """
        if not self.enabled:
            logging.warning("ByBit торговля отключена (нет API ключей)")
            return None
        
        try:
            # Получаем текущую цену
            from bybit_api_new import get_24h_tickers, get_klines
            tickers = get_24h_tickers()
            ticker = next((t for t in tickers if t["symbol"] == symbol), None)
            
            if not ticker:
                logging.error(f"Не удалось найти тикер {symbol}")
                return None
            
            current_price = float(ticker["lastPrice"])
            
            # Получаем свечи для расчета ATR
            df = get_klines(symbol, config.TIMEFRAME_MAIN, limit=200)
            import indicators
            atr_series = indicators.atr(df, 14)
            last_atr = float(atr_series.iloc[-1])
            
            # Создаем фиктивный сигнал для теста
            from models import Signal
            test_signal = Signal(
                symbol=symbol,
                side=side,
                reason="TEST MODE - Тестовое открытие позиции",
                timeframe=config.TIMEFRAME_MAIN,
                trend_tf=config.TIMEFRAME_TREND,
                last_price=current_price,
                rsi=50.0,
                ema_fast=current_price,
                ema_slow=current_price,
                atr=last_atr,
                entry=current_price,
                sl=current_price - config.ATR_SL_MULTIPLIER * last_atr if side == "LONG" else current_price + config.ATR_SL_MULTIPLIER * last_atr,
                tp1=current_price + config.ATR_TP1_MULTIPLIER * last_atr if side == "LONG" else current_price - config.ATR_TP1_MULTIPLIER * last_atr,
                tp2=current_price + config.ATR_TP2_MULTIPLIER * last_atr if side == "LONG" else current_price - config.ATR_TP2_MULTIPLIER * last_atr,
                volume_24h=float(ticker.get("quoteVolume", 0)),
                change_24h=float(ticker.get("priceChangePercent", 0)),
                tag="TEST",
                score=100.0,
            )
            
            logging.info(f"🧪 ТЕСТОВЫЙ РЕЖИМ: Открываем тестовую позицию {side} {symbol}")
            logging.info(f"   Текущая цена: {current_price:.6g}")
            logging.info(f"   ATR: {last_atr:.6g}")
            logging.info(f"   Entry: {test_signal.entry:.6g}")
            logging.info(f"   SL: {test_signal.sl:.6g}")
            logging.info(f"   TP1: {test_signal.tp1:.6g}")
            logging.info(f"   TP2: {test_signal.tp2:.6g}")
            
            # Открываем позицию используя стандартную функцию
            result = self.place_order(test_signal, risk_percent=risk_percent)
            
            if result:
                logging.info(f"✅ ТЕСТОВАЯ ПОЗИЦИЯ ОТКРЫТА: {side} {symbol}")
                logging.info(f"   Order ID: {result.get('orderId')}")
                logging.info(f"   Quantity: {result.get('qty')}")
            else:
                logging.error(f"❌ Не удалось открыть тестовую позицию {side} {symbol}")
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при тестовом открытии позиции: {e}", exc_info=True)
            return None


# Глобальный экземпляр трейдера
_trader = None


def get_trader() -> ByBitTrader:
    """Получает глобальный экземпляр трейдера"""
    global _trader
    if _trader is None:
        _trader = ByBitTrader()
    return _trader

