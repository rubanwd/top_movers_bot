"""Модуль для работы с Telegram API"""
import re
import asyncio
import logging
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from telegram import Bot, InputFile

import config
from models import Signal


def format_signals_message(
    market_trend: str,
    gain_signals: List[Signal],
    loss_signals: List[Signal],
) -> str:
    """Форматирует сообщение для отправки в .txt файл (без HTML тегов)"""
    if not gain_signals and not loss_signals:
        return "🤖 Binance Top Movers\n\nСигналов по фильтрам пока нет. Рынок спит."

    lines = []
    lines.append("🤖 Binance Futures Top Movers Signals")
    lines.append("")
    lines.append(f"🌍 Market (BTC {config.TIMEFRAME_TREND}): {market_trend}")
    lines.append("")

    def fmt_sig(sig: Signal) -> str:
        rr1 = abs((sig.tp1 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0
        rr2 = abs((sig.tp2 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0

        return (
            f"{sig.symbol} {sig.side}\n"
            f"\n"
            f"⭐ Score: {sig.score:.1f}/100\n"
            f"🏷 Tag: {sig.tag}\n"
            f"\n"
            f"📊 24h Change: {sig.change_24h:+.2f}%\n"
            f"💰 Price: {sig.last_price:.6g}\n"
            f"\n"
            f"📈 Indicators:\n"
            f"  RSI14: {sig.rsi:.1f}\n"
            f"  EMA12: {sig.ema_fast:.6g}\n"
            f"  EMA26: {sig.ema_slow:.6g}\n"
            f"  ATR14: {sig.atr:.6g}\n"
            f"\n"
            f"🎯 Levels:\n"
            f"  Entry: {sig.entry:.6g}\n"
            f"  SL: {sig.sl:.6g}\n"
            f"  TP1: {sig.tp1:.6g} (RR≈{rr1:.1f})\n"
            f"  TP2: {sig.tp2:.6g} (RR≈{rr2:.1f})\n"
        )

    if gain_signals:
        lines.append("📈 LONG candidates (Top Gainers)")
        lines.append("")
        for sig in gain_signals:
            lines.append(fmt_sig(sig))
            lines.append("")
    if loss_signals:
        if gain_signals:
            lines.append("")
        lines.append("📉 SHORT candidates (Top Losers)")
        lines.append("")
        for sig in loss_signals:
            lines.append(fmt_sig(sig))
            lines.append("")

    lines.append("")
    lines.append("⚠️ Это не финансовый совет. Торгуй головой, а не печени.")

    return "\n".join(lines)


def format_signals_message_console(
    market_trend: str,
    gain_signals: List[Signal],
    loss_signals: List[Signal],
) -> str:
    """Форматирует сообщение для вывода в консоль (без HTML-тегов)"""
    if not gain_signals and not loss_signals:
        return "🤖 Binance Top Movers\n\nСигналов по фильтрам пока нет. Рынок спит."

    lines = []
    lines.append("🤖 Binance Futures Top Movers Signals")
    lines.append("")
    lines.append(f"🌍 Market (BTC {config.TIMEFRAME_TREND}): {market_trend}")
    lines.append("")

    def fmt_sig(sig: Signal) -> str:
        rr1 = abs((sig.tp1 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0
        rr2 = abs((sig.tp2 - sig.entry) / (sig.entry - sig.sl)) if sig.entry != sig.sl else 0

        return (
            f"{sig.symbol} {sig.side} ⭐{sig.score:.1f}\n"
            f"• Tag: {sig.tag}\n"
            f"• 24h Chg: {sig.change_24h:+.2f}%\n"
            f"• Price: {sig.last_price:.6g}\n"
            f"• RSI14: {sig.rsi:.1f} | EMA12: {sig.ema_fast:.6g} | EMA26: {sig.ema_slow:.6g}\n"
            f"• ATR14: {sig.atr:.6g}\n"
            f"• Entry: {sig.entry:.6g}\n"
            f"• SL: {sig.sl:.6g}\n"
            f"• TP1: {sig.tp1:.6g} (RR≈{rr1:.1f})\n"
            f"• TP2: {sig.tp2:.6g} (RR≈{rr2:.1f})\n"
        )

    if gain_signals:
        lines.append("📈 LONG candidates (Top Gainers)")
        for sig in gain_signals:
            lines.append(fmt_sig(sig))
    if loss_signals:
        lines.append("")
        lines.append("📉 SHORT candidates (Top Losers)")
        for sig in loss_signals:
            lines.append(fmt_sig(sig))

    lines.append("")
    lines.append("⚠️ Это не финансовый совет. Торгуй головой, а не печени.")

    return "\n".join(lines)


def format_logs_message(logs: List[str], gainers: List[Dict], losers: List[Dict]) -> str:
    """Форматирует логи итерации для отправки в .txt файл"""
    lines = []
    lines.append("Логи итерации")
    lines.append("")
    
    # Основная информация
    for log in logs:
        # Убираем символы типа ✓ и другие спецсимволы
        log_line = log.replace("✓", "").strip()
        if log_line:
            lines.append(log_line)
            lines.append("")
    
    # Добавляем список топ-пар
    lines.append("")
    lines.append("=" * 50)
    lines.append("Топ Gainers (растущие монеты):")
    lines.append("")
    if gainers:
        for i, row in enumerate(gainers, 1):
            symbol = row.get("symbol", "")
            change = float(row.get("priceChangePercent", 0))
            volume = float(row.get("quoteVolume", 0))
            lines.append(f"{i}. {symbol}")
            lines.append(f"   Изменение: {change:+.2f}%")
            lines.append(f"   Объем 24h: ${volume:,.0f}")
            lines.append("")
    else:
        lines.append("Нет данных")
        lines.append("")
    
    lines.append("=" * 50)
    lines.append("Топ Losers (падающие монеты):")
    lines.append("")
    if losers:
        for i, row in enumerate(losers, 1):
            symbol = row.get("symbol", "")
            change = float(row.get("priceChangePercent", 0))
            volume = float(row.get("quoteVolume", 0))
            lines.append(f"{i}. {symbol}")
            lines.append(f"   Изменение: {change:+.2f}%")
            lines.append(f"   Объем 24h: ${volume:,.0f}")
            lines.append("")
    else:
        lines.append("Нет данных")
        lines.append("")
    
    return "\n".join(lines)


async def send_telegram_file_async(content: str, filename: str, chat_id: str, max_retries: int = 3):
    """Асинхронная отправка файла в Telegram с обработкой Flood control
    content: содержимое файла
    filename: имя файла
    chat_id: ID чата для отправки (строка или число, будет преобразовано в строку)
    max_retries: максимальное количество попыток при Flood control
    """
    # Убеждаемся, что chat_id - строка
    chat_id_str = str(chat_id).strip()
    if not chat_id_str:
        raise ValueError(f"chat_id не может быть пустым: {chat_id}")
    
    # Пересоздаем bot объект для каждой отправки, чтобы избежать проблем с event loop
    async_bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    
    try:
        # Создаем файл в памяти для каждого отправления
        file_obj = BytesIO(content.encode('utf-8'))
        file_obj.name = filename
        
        # Используем InputFile для правильной отправки файла
        input_file = InputFile(file_obj, filename=filename)
        
        logging.info(f"Отправляем файл {filename} в chat_id {chat_id_str}")
        
        # Попытки отправки с обработкой Flood control
        for attempt in range(max_retries):
            try:
                await async_bot.send_document(
                    chat_id=chat_id_str,
                    document=input_file,
                    caption=filename.replace('.txt', '').replace('_', ' ').title()
                )
                logging.info(f"Файл {filename} успешно отправлен в chat_id {chat_id_str}")
                return  # Успешно отправлено
            except Exception as e:
                error_str = str(e)
                # Проверяем, является ли это ошибкой Flood control
                if "Flood control" in error_str or "429" in error_str:
                    # Извлекаем время ожидания из сообщения об ошибке
                    retry_after = 2  # По умолчанию 2 секунды
                    if "Retry in" in error_str:
                        try:
                            # Пытаемся извлечь число из сообщения
                            match = re.search(r'Retry in (\d+)', error_str)
                            if match:
                                retry_after = int(match.group(1)) + 1
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        logging.warning(f"Flood control для {filename} в {chat_id_str}. Ожидание {retry_after} секунд перед повторной попыткой {attempt + 2}/{max_retries}")
                        await asyncio.sleep(retry_after)
                        # Пересоздаем файл для новой попытки
                        file_obj = BytesIO(content.encode('utf-8'))
                        file_obj.name = filename
                        input_file = InputFile(file_obj, filename=filename)
                        continue
                    else:
                        logging.error(f"Превышено максимальное количество попыток для {filename} в {chat_id_str} из-за Flood control")
                        raise
                else:
                    # Другая ошибка - пробрасываем сразу
                    raise
        
    except Exception as e:
        logging.error(f"Ошибка при отправке файла {filename} в chat_id {chat_id_str}: {e}", exc_info=True)
        raise
    finally:
        # Закрываем сессию bot объекта (игнорируем ошибки Flood control при закрытии)
        try:
            await async_bot.close()
        except Exception as e:
            error_str = str(e)
            if "Flood control" in error_str or "429" in error_str:
                # Это не критично - файл уже отправлен
                logging.debug(f"Flood control при закрытии bot объекта (не критично): {e}")
            else:
                logging.warning(f"Ошибка при закрытии bot объекта: {e}")


async def send_telegram_files_async(files: List[Tuple[str, str]], chat_ids: List[str]):
    """Асинхронная отправка нескольких файлов в Telegram в несколько каналов
    files: список кортежей (content, filename)
    chat_ids: список ID чатов для отправки
    """
    logging.info(f"Начинаем отправку {len(files)} файлов в {len(chat_ids)} каналов")
    errors = []
    for chat_idx, chat_id in enumerate(chat_ids):
        logging.info(f"Обрабатываем канал: {chat_id}")
        for file_idx, (content, filename) in enumerate(files):
            try:
                logging.info(f"Попытка отправить файл {filename} в канал {chat_id}")
                await send_telegram_file_async(content, filename, chat_id)
                logging.info(f"Успешно отправлен файл {filename} в канал {chat_id}")
                
                # Добавляем небольшую задержку между отправками, чтобы избежать Flood control
                # Задержка только между разными каналами или файлами (не после последнего)
                is_last_file = (file_idx == len(files) - 1)
                is_last_chat = (chat_idx == len(chat_ids) - 1)
                if not (is_last_file and is_last_chat):
                    await asyncio.sleep(0.5)  # 500ms задержка между отправками
            except Exception as e:
                error_msg = f"Не удалось отправить {filename} в {chat_id}: {e}"
                logging.error(error_msg, exc_info=True)
                errors.append(error_msg)
                # Продолжаем отправку остальных файлов даже если один не удался
                continue
    
    if errors:
        logging.warning(f"Было {len(errors)} ошибок при отправке файлов: {errors}")
        # Если все отправки провалились, пробрасываем ошибку
        if len(errors) == len(files) * len(chat_ids):
            raise RuntimeError(f"Все попытки отправки провалились: {errors}")


def send_telegram_file(content: str, filename: str, chat_ids: Optional[List[str]] = None):
    """Синхронная обертка для отправки файла в Telegram
    chat_ids: список ID чатов (по умолчанию только основной канал)
    """
    if chat_ids is None:
        chat_ids = [config.TELEGRAM_CHAT_ID]
    send_telegram_files([(content, filename)], chat_ids)


def send_telegram_files(files: List[Tuple[str, str]], chat_ids: Optional[List[str]] = None):
    """Синхронная обертка для отправки нескольких файлов в Telegram в одном event loop
    files: список кортежей (content, filename)
    chat_ids: список ID чатов для отправки (по умолчанию только основной канал)
    """
    if not files:
        logging.warning("send_telegram_files вызвана с пустым списком файлов")
        return
    
    if chat_ids is None:
        chat_ids = [config.TELEGRAM_CHAT_ID]
    
    if not chat_ids:
        logging.warning("send_telegram_files вызвана с пустым списком chat_ids")
        return
    
    try:
        logging.info(f"Вызываем asyncio.run для отправки {len(files)} файлов в {len(chat_ids)} каналов")
        # Используем asyncio.run() для правильного управления event loop
        # Это работает корректно даже при повторных вызовах из синхронного контекста
        # asyncio.run() автоматически создает новый event loop и правильно его закрывает
        asyncio.run(send_telegram_files_async(files, chat_ids))
        logging.info("asyncio.run завершился успешно")
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logging.error("Попытка вызвать asyncio.run() из запущенного event loop. Это не должно происходить в синхронном контексте.")
        logging.error("RuntimeError при отправке файлов в Telegram: %s", e, exc_info=True)
        raise
    except Exception as e:
        logging.error("Ошибка при отправке файлов в Telegram: %s", e, exc_info=True)
        raise

