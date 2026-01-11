"""
Модуль для автоматической торговли на ByBit (demo/testnet/mainnet)
"""

import logging
import time
from typing import Optional

import config
from models import Signal
from bybit_api_new import BybitAPI


class ByBitTrader:
    """Класс для работы с ByBit API для открытия позиций"""

    def __init__(self):
        self.api_key = getattr(config, "BYBIT_API_KEY", "")
        self.api_secret = getattr(config, "BYBIT_API_SECRET", "")
        self.base_url = getattr(config, "BYBIT_FUTURES_BASE", "")

        self.enabled = bool(self.api_key and self.api_secret)
        self.api: Optional[BybitAPI] = None

        # По умолчанию считаем one-way, но сразу определим фактический
        self.position_mode = "one_way"  # "one_way" | "hedge"

        if not self.enabled:
            logging.warning("ByBit API ключи не заданы. Торговля отключена.")
            return

        self.api = BybitAPI(base_url=self.base_url, api_key=self.api_key, api_secret=self.api_secret)

        # Пытаемся переключить в one-way (если разрешено на аккаунте)
        # Если не получилось — определяем текущий режим и работаем с ним.
        try:
            self.api.set_position_mode(category="linear", mode="one_way", coin="USDT")
            self.position_mode = "one_way"
            logging.info("✅ Режим позиции переключён: one-way (coin=USDT)")
        except Exception as e:
            logging.info(f"ℹ️ Не удалось переключить режим позиции через API: {e}")
            self.position_mode = self._detect_position_mode()
            logging.info(f"ℹ️ Будем работать в режиме: {self.position_mode}")

    # -------------------- Mode helpers --------------------

    def _detect_position_mode(self) -> str:
        """
        Определяем режим по данным /v5/position/list.
        Эвристика:
        - если встречаем positionIdx 1 или 2, либо несколько записей — hedge
        - иначе one_way
        """
        if not self.api:
            return "one_way"

        try:
            resp = self.api.get_positions(category="linear", symbol="BTCUSDT")
            lst = (resp.get("result", {}) or {}).get("list", []) or []
            idxs = {str(p.get("positionIdx", "")) for p in lst if p.get("positionIdx") is not None}

            if "1" in idxs or "2" in idxs or len(lst) >= 2:
                return "hedge"
            return "one_way"
        except Exception:
            return "one_way"

    def _position_idx(self, side: str) -> int:
        """
        side: "Buy" | "Sell"
        """
        if self.position_mode == "hedge":
            return 1 if side == "Buy" else 2
        return 0

    # -------------------- Balance helpers --------------------

    def get_account_info(self) -> dict:
        if not self.enabled or not self.api:
            raise RuntimeError("ByBit API ключи не настроены")
        return self.api.get_account_info()

    def get_symbol_info(self, symbol: str) -> dict:
        if not self.enabled or not self.api:
            raise RuntimeError("ByBit API ключи не настроены")
        return self.api.get_symbol_info(symbol)

    def _get_usdt_balance(self) -> float:
        account_info = self.get_account_info()
        balance_list = (account_info.get("result", {}) or {}).get("list", []) or []
        for account in balance_list:
            for coin in account.get("coin", []) or []:
                if coin.get("coin") == "USDT":
                    return float(coin.get("walletBalance", 0))
        return 0.0

    # -------------------- Position sizing --------------------

    def calculate_position_size(self, symbol: str, entry_price: float, sl_price: float, risk_percent: float = 1.0) -> float:
        """
        Рассчитывает размер позиции по риску (ATR-стоп у тебя уже приходит в signal.sl)
        """
        try:
            total_equity = self._get_usdt_balance()

            if total_equity <= 0:
                logging.warning(f"❌ Баланс USDT равен 0 для {symbol}")
                return 0.0

            logging.info(f"Баланс USDT: {total_equity:.2f}")

            risk_amount = total_equity * (risk_percent / 100.0)
            logging.info(f"Риск на сделку: ${risk_amount:.2f} ({risk_percent}% от баланса ${total_equity:.2f})")

            risk_per_contract = abs(entry_price - sl_price)
            if risk_per_contract <= 0:
                logging.warning(f"❌ Риск на контракт равен 0 для {symbol} (entry={entry_price}, SL={sl_price})")
                return 0.0

            logging.info(f"Риск на контракт: {risk_per_contract:.6g}")

            symbol_info = self.get_symbol_info(symbol)
            instruments = (symbol_info.get("result", {}) or {}).get("list", []) or []
            if not instruments:
                logging.warning(f"Не удалось получить информацию о символе {symbol}")
                return 0.0

            instrument = instruments[0]
            lot_size_filter = instrument.get("lotSizeFilter", {}) or {}
            qty_step = float(lot_size_filter.get("qtyStep", "1"))
            min_qty = float(lot_size_filter.get("minQty", "0"))

            qty = risk_amount / risk_per_contract
            logging.info(f"Предварительный размер позиции: {qty:.6g} контрактов")

            # округление
            qty = round(qty / qty_step) * qty_step
            logging.info(f"После округления до шага {qty_step}: {qty:.6g}")

            if min_qty > 0 and qty < min_qty:
                logging.info(f"Размер позиции меньше минимума ({qty:.6g} < {min_qty}), ставим минимум")
                qty = min_qty

            if qty <= 0:
                logging.warning(f"❌ Итоговый размер позиции равен 0 для {symbol}")
                return 0.0

            return qty

        except Exception as e:
            logging.error(f"Ошибка при расчете размера позиции: {e}", exc_info=True)
            return 0.0

    # -------------------- Trading --------------------

    def _ensure_leverage(self, symbol: str) -> int:
        """
        Ставит плечо, если возможно.
        Берёт config.LEVERAGE (если есть), иначе 10.
        """
        if not self.api:
            return 1

        leverage = int(getattr(config, "LEVERAGE", 10) or 10)

        try:
            self.api.set_leverage(category="linear", symbol=symbol, buy_leverage=leverage, sell_leverage=leverage)
            logging.info(f"✅ Leverage установлен {leverage}x для {symbol}")
        except Exception as e:
            logging.info(f"ℹ️ Не удалось установить leverage для {symbol}: {e}")

        return leverage

    def _cap_qty_by_margin(self, qty: float, entry_price: float, leverage: int) -> float:
        """
        Ограничиваем qty по доступной марже, чтобы не ловить ab not enough.
        """
        if qty <= 0:
            return 0.0

        balance = self._get_usdt_balance()
        if balance <= 0:
            return 0.0

        # Простая оценка: required_margin ~ notional / leverage
        notional = qty * float(entry_price)
        required_margin = notional / max(leverage, 1)

        max_margin = balance * float(getattr(config, "MARGIN_UTILIZATION", 0.95) or 0.95)

        if required_margin > max_margin and required_margin > 0:
            scale = max_margin / required_margin
            new_qty = qty * scale
            logging.info(
                f"⚠️ Ограничили qty по марже: было {qty:.6g}, стало {new_qty:.6g} "
                f"(баланс={balance:.2f}, lev={leverage}x, req={required_margin:.2f}, max={max_margin:.2f})"
            )
            return new_qty

        return qty

    def has_open_position(self, symbol: str) -> bool:
        """
        Проверяет, есть ли уже открытая позиция по символу
        """
        if not self.enabled or not self.api:
            return False
        
        try:
            resp = self.api.get_positions(category="linear", symbol=symbol)
            positions = (resp.get("result", {}) or {}).get("list", []) or []
            
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                if size != 0:
                    return True
            return False
        except Exception as e:
            logging.warning(f"Не удалось проверить открытые позиции для {symbol}: {e}")
            # В случае ошибки разрешаем открытие (безопаснее пропустить проверку, чем блокировать все сделки)
            return False

    def place_order(self, signal: Signal, qty: Optional[float] = None, risk_percent: float = 1.0) -> Optional[dict]:
        """
        Открывает позицию на основе сигнала
        """
        if not self.enabled or not self.api:
            logging.warning("ByBit торговля отключена (нет API ключей)")
            return None

        # Проверяем, нет ли уже открытой позиции по этому символу
        if self.has_open_position(signal.symbol):
            logging.info(f"⏭️ Пропускаем {signal.symbol} - уже есть открытая позиция")
            return None

        try:
            side = "Buy" if signal.side == "LONG" else "Sell"
            pos_idx = self._position_idx(side)

            # Установим плечо
            leverage = self._ensure_leverage(signal.symbol)

            # Рассчитываем qty, если не задан
            if qty is None:
                logging.info(
                    f"Рассчитываем размер позиции для {signal.symbol} "
                    f"(риск {risk_percent}%, entry={signal.entry:.6g}, SL={signal.sl:.6g})"
                )
                qty = self.calculate_position_size(signal.symbol, signal.entry, signal.sl, risk_percent)
                logging.info(f"Рассчитанный размер позиции для {signal.symbol}: {qty}")

            if not qty or qty <= 0:
                logging.warning(f"❌ Qty для {signal.symbol} равен 0 — пропускаем")
                return None

            # Ограничим по марже
            qty = self._cap_qty_by_margin(qty, signal.entry, leverage)
            if qty <= 0:
                logging.warning(f"❌ Qty после ограничения по марже стал 0 — пропускаем {signal.symbol}")
                return None

            # Округлим qty под шаг лота ещё раз (после cap)
            symbol_info = self.get_symbol_info(signal.symbol)
            instruments = (symbol_info.get("result", {}) or {}).get("list", []) or []
            if instruments:
                lot_size_filter = instruments[0].get("lotSizeFilter", {}) or {}
                qty_step = float(lot_size_filter.get("qtyStep", "1"))
                qty = round(qty / qty_step) * qty_step

            order_params = {
                "category": "linear",
                "symbol": signal.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "positionIdx": pos_idx,
            }

            order_result = self.api.place_order(**order_params)
            result_data = order_result.get("result", {}) or {}
            order_id = result_data.get("orderId")

            if not order_id:
                logging.error(f"Не удалось получить orderId для {signal.symbol}")
                return None

            logging.info(f"✅ Открыта позиция {signal.side} {signal.symbol}: qty={qty}, orderId={order_id}, posIdx={pos_idx}")

            sl_ok = self._set_stop_loss(signal, pos_idx)
            tp_ok = self._set_take_profits(signal, pos_idx)

            return {
                "orderId": order_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "qty": float(qty),
                "entry": signal.entry,
                "sl": signal.sl,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "positionIdx": pos_idx,
                "sl_set": sl_ok,
                "tp_set": tp_ok,
            }

        except Exception as e:
            logging.error(f"Ошибка при открытии позиции для {signal.symbol}: {e}", exc_info=True)
            return None

    def _set_stop_loss(self, signal: Signal, pos_idx: int) -> bool:
        if not self.enabled or not self.api:
            return False

        try:
            res = self.api.set_sl_tp(
                category="linear",
                symbol=signal.symbol,
                positionIdx=pos_idx,
                stopLoss=str(signal.sl),
            )
            logging.info(f"✅ SL установлен для {signal.symbol}: {signal.sl} (posIdx={pos_idx})")
            return bool(res)
        except Exception as e:
            logging.error(f"Ошибка при установке SL для {signal.symbol}: {e}", exc_info=True)
            return False

    def _set_take_profits(self, signal: Signal, pos_idx: int) -> bool:
        """
        Bybit поддерживает 1 TP через trading-stop.
        Ставим TP1.
        """
        if not self.enabled or not self.api:
            return False

        try:
            time.sleep(0.5)
            res = self.api.set_sl_tp(
                category="linear",
                symbol=signal.symbol,
                positionIdx=pos_idx,
                takeProfit=str(signal.tp1),
            )
            logging.info(f"✅ TP1 установлен для {signal.symbol}: {signal.tp1} (posIdx={pos_idx})")
            return bool(res)
        except Exception as e:
            logging.error(f"Ошибка при установке TP для {signal.symbol}: {e}", exc_info=True)
            return False

    # -------------------- Test helper --------------------

    def test_order_placement(self, symbol: str = "BTCUSDT", side: str = "LONG", risk_percent: float = 1.0) -> Optional[dict]:
        if not self.enabled:
            logging.warning("ByBit торговля отключена (нет API ключей)")
            return None

        try:
            from bybit_api_new import get_24h_tickers, get_klines
            import indicators

            tickers = get_24h_tickers()
            ticker = next((t for t in tickers if t["symbol"] == symbol), None)
            if not ticker:
                logging.error(f"Не удалось найти тикер {symbol}")
                return None

            current_price = float(ticker["lastPrice"])
            df = get_klines(symbol, config.TIMEFRAME_MAIN, limit=200)
            atr_series = indicators.atr(df, 14)
            last_atr = float(atr_series.iloc[-1])

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
            logging.info(f"   Цена: {current_price:.6g} | ATR: {last_atr:.6g}")
            logging.info(f"   Entry: {test_signal.entry:.6g} | SL: {test_signal.sl:.6g}")
            logging.info(f"   TP1: {test_signal.tp1:.6g} | TP2: {test_signal.tp2:.6g}")
            logging.info(f"   Режим позиций: {self.position_mode}")

            result = self.place_order(test_signal, risk_percent=risk_percent)
            if result:
                logging.info(f"✅ ТЕСТОВАЯ ПОЗИЦИЯ ОТКРЫТА: {side} {symbol} | {result}")
            else:
                logging.error(f"❌ Не удалось открыть тестовую позицию {side} {symbol}")

            return result

        except Exception as e:
            logging.error(f"Ошибка при тестовом открытии позиции: {e}", exc_info=True)
            return None


# Глобальный экземпляр трейдера
_trader = None


def get_trader() -> ByBitTrader:
    global _trader
    if _trader is None:
        _trader = ByBitTrader()
    return _trader
