"""Основной модуль бота для отслеживания топ-муверов на ByBit/Binance Futures"""
import time
import logging
from datetime import datetime
from typing import List, Dict

import config
import signal_logic
import telegram_handler
from models import Signal

# Импортируем API в зависимости от выбранной биржи
if config.EXCHANGE == "bybit":
    import bybit_api as exchange_api
    import bybit_trading
else:
    import binance_api as exchange_api
    # Для Binance торговля пока не реализована
    bybit_trading = None

# Настройка логирования уже выполнена в config.py


def run_once():
    """Выполняет одну итерацию сканирования и отправки сигналов"""
    # Собираем логи итерации
    iteration_logs: List[str] = []
    
    logging.info("Старт сканирования Top Movers...")
    iteration_logs.append("Старт сканирования Top Movers...")
    
    tickers = exchange_api.get_24h_tickers()
    gainers, losers = signal_logic.select_top_movers(tickers)
    logging.info("Отобрано gainers=%d, losers=%d", len(gainers), len(losers))
    iteration_logs.append(f"Отобрано gainers={len(gainers)}, losers={len(losers)}")

    market_trend = signal_logic.detect_market_trend_btc()
    logging.info("BTC market trend: %s", market_trend)
    iteration_logs.append(f"BTC market trend: {market_trend}")

    gain_signals: List[Signal] = []
    loss_signals: List[Signal] = []

    checked_symbols = []
    for row in gainers:
        sym = row["symbol"]
        logging.info("Проверяем LONG %s", sym)
        checked_symbols.append(f"LONG {sym}")
        sig = signal_logic.build_signal(sym, "LONG", row, market_trend)
        if sig:
            gain_signals.append(sig)
            iteration_logs.append(f"LONG {sym} - сигнал найден (score: {sig.score:.1f})")

    for row in losers:
        sym = row["symbol"]
        logging.info("Проверяем SHORT %s", sym)
        checked_symbols.append(f"SHORT {sym}")
        sig = signal_logic.build_signal(sym, "SHORT", row, market_trend)
        if sig:
            loss_signals.append(sig)
            iteration_logs.append(f"SHORT {sym} - сигнал найден (score: {sig.score:.1f})")
    
    # Сортируем сигналы по оценке качества и ограничиваем количество
    gain_signals.sort(key=lambda x: x.score, reverse=True)
    loss_signals.sort(key=lambda x: x.score, reverse=True)
    
    # Ограничиваем количество сигналов (максимум MAX_SIGNALS_PER_DAY в день)
    # Распределяем поровну между LONG и SHORT, но приоритет лучшим
    max_per_side = config.MAX_SIGNALS_PER_DAY // 2
    if len(gain_signals) > max_per_side:
        gain_signals = gain_signals[:max_per_side]
    if len(loss_signals) > max_per_side:
        loss_signals = loss_signals[:max_per_side]
    
    # Если одна сторона пустая, можно дать больше другой (но не больше MAX_SIGNALS_PER_DAY)
    total_signals = len(gain_signals) + len(loss_signals)
    if total_signals > config.MAX_SIGNALS_PER_DAY:
        # Урезаем худшие сигналы
        if len(gain_signals) > len(loss_signals):
            gain_signals = gain_signals[:config.MAX_SIGNALS_PER_DAY - len(loss_signals)]
        else:
            loss_signals = loss_signals[:config.MAX_SIGNALS_PER_DAY - len(gain_signals)]
    
    iteration_logs.append(f"Проверено символов: {len(checked_symbols)}")
    iteration_logs.append(f"Найдено сигналов: LONG={len(gain_signals)}, SHORT={len(loss_signals)} (после фильтрации)")

    msg = telegram_handler.format_signals_message(market_trend, gain_signals, loss_signals)
    msg_console = telegram_handler.format_signals_message_console(market_trend, gain_signals, loss_signals)
    
    # Вывод в консоль
    print("\n" + "=" * 80)
    print(msg_console)
    print("=" * 80 + "\n")
    
    try:
        # Отправляем файлы в одном event loop
        logs_msg = telegram_handler.format_logs_message(iteration_logs, gainers, losers)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_to_send = []
        signals_files_to_send = []
        
        # Отправляем файл с сигналами только если есть хотя бы один сигнал
        if gain_signals or loss_signals:
            signals_files_to_send.append((msg, f"signals_{timestamp}.txt"))
            logging.info("Найдены сигналы, будет отправлен файл signals.")
        else:
            logging.info("Сигналов не найдено, файл signals не отправляется.")
        
        # Логи отправляем всегда в основной канал
        files_to_send.append((logs_msg, f"logs_{timestamp}.txt"))
        
        # Отправляем логи в основной канал
        if files_to_send:
            telegram_handler.send_telegram_files(files_to_send, [config.TELEGRAM_CHAT_ID])
            logging.info("Отправлены файлы логов в основной канал Telegram.")
        
        # Если есть сигналы, отправляем их в оба канала (основной и дополнительный)
        if signals_files_to_send:
            chat_ids = [config.TELEGRAM_CHAT_ID]
            if config.TELEGRAM_CHAT_ID_2:
                chat_ids.append(config.TELEGRAM_CHAT_ID_2)
                logging.info(f"Отправляем сигналы в основной ({config.TELEGRAM_CHAT_ID}) и дополнительный ({config.TELEGRAM_CHAT_ID_2}) каналы.")
            else:
                logging.info(f"Отправляем сигналы только в основной канал ({config.TELEGRAM_CHAT_ID}, дополнительный не задан).")
            
            logging.info(f"Вызываем send_telegram_files с {len(signals_files_to_send)} файлами и {len(chat_ids)} каналами")
            try:
                telegram_handler.send_telegram_files(signals_files_to_send, chat_ids)
                logging.info("Отправлены файлы сигналов в Telegram.")
                
                # Открываем позиции на бирже для каждого сигнала (только для ByBit)
                if config.EXCHANGE == "bybit" and config.BYBIT_ENABLE_TRADING:
                    trader = bybit_trading.get_trader()
                    if trader.enabled:
                        all_signals = gain_signals + loss_signals
                        for signal in all_signals:
                            try:
                                result = trader.place_order(signal, risk_percent=config.BYBIT_RISK_PERCENT)
                                if result:
                                    logging.info(f"✅ Позиция открыта: {signal.side} {signal.symbol} | Entry: {signal.entry:.6g} | SL: {signal.sl:.6g} | TP1: {signal.tp1:.6g} | TP2: {signal.tp2:.6g}")
                                    iteration_logs.append(f"✅ Позиция открыта: {signal.side} {signal.symbol}")
                                else:
                                    logging.warning(f"❌ Не удалось открыть позицию для {signal.side} {signal.symbol}")
                                    iteration_logs.append(f"❌ Не удалось открыть позицию: {signal.side} {signal.symbol}")
                            except Exception as e:
                                logging.error(f"Ошибка при открытии позиции для {signal.symbol}: {e}", exc_info=True)
                                iteration_logs.append(f"❌ Ошибка открытия позиции {signal.symbol}: {str(e)}")
                    else:
                        logging.warning("ByBit торговля отключена (нет API ключей)")
                elif config.EXCHANGE == "bybit" and not config.BYBIT_ENABLE_TRADING:
                    logging.info("ByBit торговля отключена в настройках (BYBIT_ENABLE_TRADING=0)")
                elif config.EXCHANGE == "binance":
                    logging.info("Автоматическая торговля для Binance пока не реализована")
            except Exception as e:
                logging.error(f"КРИТИЧЕСКАЯ ОШИБКА при отправке сигналов: {e}", exc_info=True)
    except Exception as e:
        logging.warning("Не удалось отправить файлы в Telegram: %s", e)


def main():
    """Главная функция - запускает бесконечный цикл сканирования"""
    exchange_name = config.EXCHANGE.upper()
    logging.info(f"Запускаем {exchange_name} Top Movers bot (FULL mode).")
    logging.info(f"Текущие настройки: TOP_N={config.TOP_N}, SCAN_INTERVAL_SECONDS={config.SCAN_INTERVAL_SECONDS}")
    while True:
        try:
            run_once()
        except Exception as e:
            logging.exception("Ошибка в run_once: %s", e)
            try:
                error_msg = f"⚠️ Бот поймал ошибку:\n{str(e)}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                telegram_handler.send_telegram_file(error_msg, f"error_{timestamp}.txt")
            except Exception as telegram_error:
                logging.warning("Не удалось отправить ошибку в Telegram: %s", telegram_error)
        logging.info("Спим %d секунд...", config.SCAN_INTERVAL_SECONDS)
        time.sleep(config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
