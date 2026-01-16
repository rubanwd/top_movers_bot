"""Модуль логики генерации торговых сигналов"""
import math
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional

import config
import indicators
from models import Signal

# Импортируем API в зависимости от выбранной биржи
if config.EXCHANGE == "bybit":
    import bybit_api as exchange_api
else:
    import binance_api as exchange_api


def select_top_movers(tickers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Выбирает топ растущих и падающих монет"""
    df = pd.DataFrame(tickers)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    df["quoteVolume"] = df["quoteVolume"].astype(float)

    df = df[df["quoteVolume"] >= config.MIN_QUOTE_VOLUME_USDT]
    
    # Фильтруем монеты, которые уже слишком сильно выросли/упали (опционально)
    df_gainers = df[df["priceChangePercent"] > 0]
    df_losers = df[df["priceChangePercent"] < 0]
    
    if config.USE_MAX_24H_FILTER and config.MAX_24H_CHANGE > 0:
        # Берем только те, где изменение за 24ч не превышает MAX_24H_CHANGE
        df_gainers = df_gainers[df_gainers["priceChangePercent"] <= config.MAX_24H_CHANGE]
        df_losers = df_losers[df_losers["priceChangePercent"] >= -config.MAX_24H_CHANGE]

    gainers = df_gainers.sort_values("priceChangePercent", ascending=False).head(config.TOP_N)
    losers = df_losers.sort_values("priceChangePercent", ascending=True).head(config.TOP_N)

    return gainers.to_dict("records"), losers.to_dict("records")


def detect_market_trend_btc() -> str:
    """Определяет общий тренд рынка по BTC"""
    try:
        df = exchange_api.get_klines("BTCUSDT", config.TIMEFRAME_TREND, limit=200)
    except Exception as e:
        logging.warning("Не удалось получить BTCUSDT для тренда: %s", e)
        return "UNKNOWN"

    close = df["close"]
    ema_fast = indicators.ema(close, 20)
    ema_slow = indicators.ema(close, 50)
    rsi_val = indicators.rsi(close, 14)

    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_rsi = float(rsi_val.iloc[-1])

    if math.isnan(last_ema_fast) or math.isnan(last_ema_slow) or math.isnan(last_rsi):
        return "UNKNOWN"

    if last_ema_fast > last_ema_slow and last_rsi > 55:
        return "UP"
    elif last_ema_fast < last_ema_slow and last_rsi < 45:
        return "DOWN"
    else:
        return "SIDE"


def calculate_signal_score(
    side: str,
    rsi_val: float,
    ema_fast_val: float,
    ema_slow_val: float,
    macd_hist: float,
    adx_val: float,
    vol_spike: bool,
    momentum_ok: bool,
    recent_change_pct: float,
    price_change_24h: float,
) -> float:
    """Вычисляет оценку качества сигнала (0-100)"""
    score = 0.0
    
    # RSI в оптимальной зоне (25 баллов) - адаптировано под сбалансированные зоны
    if side == "LONG":
        if 48 <= rsi_val <= 58:  # Оптимальная зона для входа в LONG
            score += 25
        elif 46 <= rsi_val < 48 or 58 < rsi_val <= 60:
            score += 15
        elif 44 <= rsi_val < 46 or 60 < rsi_val <= 62:
            score += 8
    else:  # SHORT
        if 42 <= rsi_val <= 50:  # Оптимальная зона для входа в SHORT
            score += 25
        elif 40 <= rsi_val < 42 or 50 < rsi_val <= 52:
            score += 15
        elif 38 <= rsi_val < 40 or 52 < rsi_val <= 54:
            score += 8
    
    # EMA пересечение и сила тренда (30 баллов) - адаптировано под сбалансированные требования
    if side == "LONG":
        if ema_fast_val > ema_slow_val:
            ema_diff_pct = ((ema_fast_val - ema_slow_val) / ema_slow_val) * 100
            # Требуем минимум 0.08% для начисления баллов
            if ema_diff_pct >= 0.08:
                score += min(30, ema_diff_pct * 4)  # Баллы за силу тренда
    else:  # SHORT
        if ema_fast_val < ema_slow_val:
            ema_diff_pct = ((ema_slow_val - ema_fast_val) / ema_slow_val) * 100
            # Требуем минимум 0.08% для начисления баллов
            if ema_diff_pct >= 0.08:
                score += min(30, ema_diff_pct * 4)  # Баллы за силу тренда
    
    # MACD подтверждение (15 баллов) - теперь опционально, не блокирует сигнал
    if config.USE_MACD:
        if side == "LONG" and macd_hist > 0:
            score += 15
        elif side == "SHORT" and macd_hist < 0:
            score += 15
        # Если MACD не подтверждает, просто не даем бонусы (не блокируем сигнал)
    
    # ADX сила тренда (15 баллов) - теперь опционально, не блокирует сигнал
    if config.USE_ADX:
        if adx_val >= config.MIN_ADX:
            score += min(15, (adx_val - config.MIN_ADX) / 2)  # До 15 баллов за сильный тренд
        # Если ADX слабый, просто не даем бонусы (не блокируем сигнал)
    
    # Объем (15 баллов) - увеличено, так как теперь обязательное условие
    if vol_spike:
        # Дополнительные баллы за очень сильный всплеск объема
        score += 15
    
    # Momentum (15 баллов) - увеличено, так как теперь обязательное условие
    if momentum_ok:
        score += 15
    
    # Недавнее движение (5 баллов)
    if abs(recent_change_pct) >= config.MIN_RECENT_CHANGE_PCT:
        score += 5
    
    return min(100.0, score)


def build_signal(symbol: str, side: str, ticker_row: Dict, market_trend: str) -> Optional[Signal]:
    """Строит торговый сигнал для символа"""
    try:
        df_main = exchange_api.get_klines(symbol, config.TIMEFRAME_MAIN, limit=200)
        _ = exchange_api.get_klines(symbol, config.TIMEFRAME_TREND, limit=200)
    except Exception as e:
        logging.warning("Klines error for %s: %s", symbol, e)
        return None

    close = df_main["close"]
    vol = df_main["volume"]

    # Используем более быстрые EMA для раннего обнаружения
    ema_fast = indicators.ema(close, 12)  # Изменено с 20 на 12 для более раннего обнаружения
    ema_slow = indicators.ema(close, 26)  # Изменено с 50 на 26 для более раннего обнаружения
    rsi_series = indicators.rsi(close, 14)
    atr_series = indicators.atr(df_main, 14)
    
    # Добавляем MACD и ADX для улучшенной фильтрации
    macd_line, macd_signal, macd_hist = indicators.macd(close, 12, 26, 9)
    adx_series = indicators.adx(df_main, 14)

    last_close = float(close.iloc[-1])
    last_rsi = float(rsi_series.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_atr = float(atr_series.iloc[-1])
    last_macd_hist = float(macd_hist.iloc[-1]) if config.USE_MACD else 0.0
    last_adx = float(adx_series.iloc[-1]) if config.USE_ADX else 0.0

    if any(math.isnan(x) for x in [last_rsi, last_ema_fast, last_ema_slow, last_atr]):
        return None
    
    if config.USE_MACD and math.isnan(last_macd_hist):
        return None
    if config.USE_ADX and math.isnan(last_adx):
        return None

    avg_vol = float(vol.iloc[-50:].mean())
    last_vol = float(vol.iloc[-1])
    vol_spike = last_vol > config.VOL_SPIKE_MULTIPLIER * avg_vol if avg_vol > 0 else False
    
    # Улучшенная проверка momentum (ускорение цены) - оптимизировано для лучшего win rate
    momentum_ok = False
    if len(close) >= 5:  # Увеличено окно для более надежного определения momentum
        # Проверяем ускорение: цена должна расти/падать быстрее
        price_change_1 = (last_close - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        price_change_2 = (float(close.iloc[-2]) - float(close.iloc[-3])) / float(close.iloc[-3]) * 100
        price_change_3 = (float(close.iloc[-3]) - float(close.iloc[-4])) / float(close.iloc[-4]) * 100
        
        if side == "LONG":
            # Для LONG: сбалансированные требования для получения нескольких сигналов в день
            # Принимаем ускорение роста ИЛИ стабильный рост ИЛИ просто положительное движение
            momentum_ok = (price_change_1 > price_change_2 and price_change_1 > 0.1) or \
                         (price_change_1 > 0.15 and price_change_2 > 0.05) or \
                         (price_change_1 > 0.12 and price_change_2 > 0.08) or \
                         (price_change_1 > 0.1 and price_change_2 > 0.1 and price_change_3 > 0.05)
        else:
            # Для SHORT: сбалансированные требования для получения нескольких сигналов в день
            # Принимаем ускорение падения ИЛИ стабильное падение ИЛИ просто отрицательное движение
            momentum_ok = (price_change_1 < price_change_2 and price_change_1 < -0.1) or \
                         (price_change_1 < -0.15 and price_change_2 < -0.05) or \
                         (price_change_1 < -0.12 and price_change_2 < -0.08) or \
                         (price_change_1 < -0.1 and price_change_2 < -0.1 and price_change_3 < -0.05)

    # ========== ПРОВЕРКИ ДЛЯ РАННЕГО ОБНАРУЖЕНИЯ ДВИЖЕНИЯ (ОПЦИОНАЛЬНЫЕ) ==========
    
    # Вычисляем недавнее изменение для тегов (всегда)
    recent_change_pct = 0.0
    if len(close) >= config.RECENT_CANDLES_LOOKBACK + 1:
        recent_start_idx = -config.RECENT_CANDLES_LOOKBACK - 1
        recent_start_price = float(close.iloc[recent_start_idx])
        recent_change_pct = ((last_close - recent_start_price) / recent_start_price) * 100
    
    # 1. Проверка недавнего изменения цены (за последние N свечей) - ОПЦИОНАЛЬНО
    recent_move_ok = True  # По умолчанию пропускаем, если проверка выключена
    if config.RECENT_MOVE_CHECK:
        if len(close) < config.RECENT_CANDLES_LOOKBACK + 1:
            logging.debug(f"{symbol} {side}: недостаточно данных для RECENT_MOVE_CHECK")
            return None
        
        # Строгая проверка: требуем значимое движение в правильном направлении
        if side == "LONG":
            # Для LONG: требуем изменение >= минимума
            recent_move_ok = recent_change_pct >= config.MIN_RECENT_CHANGE_PCT
        else:
            # Для SHORT: требуем изменение <= -минимума
            recent_move_ok = recent_change_pct <= -config.MIN_RECENT_CHANGE_PCT
        
        if not recent_move_ok:
            logging.info(f"{symbol} {side}: ❌ RECENT_MOVE не прошел (изменение: {recent_change_pct:.2f}%, требуется: {config.MIN_RECENT_CHANGE_PCT}%)")
            return None
    
    # 2. Проверка, что RSI только что вошел в нужную зону - ОПЦИОНАЛЬНО (ослаблена)
    rsi_entry_ok = True
    if config.RSI_ENTRY_CHECK:
        if len(rsi_series) < 3:
            logging.debug(f"{symbol} {side}: недостаточно данных для RSI_ENTRY_CHECK")
            return None
        
        prev_rsi = float(rsi_series.iloc[-2])
        prev_prev_rsi = float(rsi_series.iloc[-3])
        
        if side == "LONG":
            # Строгая проверка: RSI в зоне И только что вошел в зону (ранний вход)
            rsi_in_zone = config.RSI_LONG_MIN <= last_rsi <= config.RSI_LONG_MAX
            rsi_just_entered = (prev_rsi < config.RSI_LONG_MIN or prev_prev_rsi < config.RSI_LONG_MIN) and rsi_in_zone
            rsi_rising = last_rsi > prev_rsi  # RSI растет
            # Принимаем если RSI в зоне И (только что вошел ИЛИ растет)
            rsi_entry_ok = rsi_in_zone and (rsi_just_entered or rsi_rising)
        else:
            # Для SHORT: строгая проверка
            rsi_in_zone = config.RSI_SHORT_MIN <= last_rsi <= config.RSI_SHORT_MAX
            rsi_just_entered = (prev_rsi > config.RSI_SHORT_MAX or prev_prev_rsi > config.RSI_SHORT_MAX) and rsi_in_zone
            rsi_falling = last_rsi < prev_rsi  # RSI падает
            # Принимаем если RSI в зоне И (только что вошел ИЛИ падает)
            rsi_entry_ok = rsi_in_zone and (rsi_just_entered or rsi_falling)
        
        if not rsi_entry_ok:
            logging.info(f"{symbol} {side}: ❌ RSI_ENTRY не прошел (RSI: {last_rsi:.1f}, prev: {prev_rsi:.1f})")
            return None
    
    # 3. Проверка недавнего пересечения EMA - ОПЦИОНАЛЬНО (ослаблена)
    ema_cross_ok = True
    if config.EMA_CROSS_RECENT:
        if len(ema_fast) < 3 or len(ema_slow) < 3:
            logging.debug(f"{symbol} {side}: недостаточно данных для EMA_CROSS_RECENT")
            return None
        
        prev_ema_fast = float(ema_fast.iloc[-2])
        prev_ema_slow = float(ema_slow.iloc[-2])
        prev_prev_ema_fast = float(ema_fast.iloc[-3]) if len(ema_fast) >= 3 else prev_ema_fast
        prev_prev_ema_slow = float(ema_slow.iloc[-3]) if len(ema_slow) >= 3 else prev_ema_slow
        
        if side == "LONG":
            # Строгая проверка: EMA в правильном порядке И недавно пересекли
            ema_correct_order = last_ema_fast > last_ema_slow
            ema_crossed = (prev_ema_fast <= prev_ema_slow or prev_prev_ema_fast <= prev_prev_ema_slow) and ema_correct_order
            # Принимаем только если EMA недавно пересекли (ранний вход в тренд)
            ema_cross_ok = ema_correct_order and ema_crossed
        else:
            # Для SHORT: строгая проверка
            ema_correct_order = last_ema_fast < last_ema_slow
            ema_crossed = (prev_ema_fast >= prev_ema_slow or prev_prev_ema_fast >= prev_prev_ema_slow) and ema_correct_order
            # Принимаем только если EMA недавно пересекли (ранний вход в тренд)
            ema_cross_ok = ema_correct_order and ema_crossed
        
        if not ema_cross_ok:
            logging.info(f"{symbol} {side}: ❌ EMA_CROSS не прошел (EMA12: {last_ema_fast:.6g}, EMA26: {last_ema_slow:.6g})")
            return None
    
    # 4. Проверка, что объем начал расти недавно - ОБЯЗАТЕЛЬНО для качества
    vol_recent_ok = True
    if config.VOL_RECENT_CHECK:
        if len(vol) >= config.RECENT_CANDLES_LOOKBACK:
            recent_vols = vol.iloc[-config.RECENT_CANDLES_LOOKBACK:].astype(float)
            recent_avg_vol = float(recent_vols.mean())
            # Объем за последние N свечей должен быть выше среднего (сбалансировано)
            vol_recent_ok = recent_avg_vol > avg_vol * 1.08  # 8% выше среднего
        
        if not vol_recent_ok:
            logging.info(f"{symbol} {side}: ❌ VOL_RECENT не прошел (recent_avg: {recent_avg_vol:.0f}, avg: {avg_vol:.0f})")
            return None
    
    # ========== ОСНОВНЫЕ ПРОВЕРКИ ==========
    
    trend_ok = False
    rsi_ok = False

    if side == "LONG":
        # Сбалансированная проверка тренда: требуем минимальную разницу EMA
        ema_diff_pct = ((last_ema_fast - last_ema_slow) / last_ema_slow) * 100
        # Требуем минимум 0.08% разницы для подтверждения направления тренда
        trend_ok = last_ema_fast > last_ema_slow and ema_diff_pct >= 0.08
        rsi_ok = config.RSI_LONG_MIN <= last_rsi <= config.RSI_LONG_MAX
    else:
        # Сбалансированная проверка тренда: требуем минимальную разницу EMA
        ema_diff_pct = ((last_ema_slow - last_ema_fast) / last_ema_slow) * 100
        # Требуем минимум 0.08% разницы для подтверждения направления тренда
        trend_ok = last_ema_fast < last_ema_slow and ema_diff_pct >= 0.08
        rsi_ok = config.RSI_SHORT_MIN <= last_rsi <= config.RSI_SHORT_MAX

    # Дополнительная проверка: минимальный объем за 24ч должен быть достаточно большим
    volume_24h_ok = float(ticker_row["quoteVolume"]) >= config.MIN_QUOTE_VOLUME_USDT
    
    if not volume_24h_ok:
        logging.info(f"{symbol} {side}: ❌ объем 24ч слишком мал ({float(ticker_row['quoteVolume']):,.0f} < {config.MIN_QUOTE_VOLUME_USDT:,.0f})")
        return None
    
    # Дополнительный фильтр: избегаем входов когда цена слишком далеко от EMA
    # Сбалансированный фильтр: цена должна быть не слишком далеко от EMA
    if side == "LONG":
        price_to_ema_fast = ((last_close - last_ema_fast) / last_ema_fast) * 100
        # Если цена более чем на 2.0% выше быстрой EMA, возможно уже поздно входить
        if price_to_ema_fast > 2.0:
            logging.info(f"{symbol} {side}: ❌ цена слишком далеко от EMA быстрой ({price_to_ema_fast:.2f}% выше)")
            return None
    else:  # SHORT
        price_to_ema_fast = ((last_ema_fast - last_close) / last_ema_fast) * 100
        # Если цена более чем на 2.0% ниже быстрой EMA, возможно уже поздно входить
        if price_to_ema_fast > 2.0:
            logging.info(f"{symbol} {side}: ❌ цена слишком далеко от EMA быстрой ({price_to_ema_fast:.2f}% ниже)")
            return None
    
    # Основные проверки: тренд, RSI, и (всплеск объема ИЛИ momentum) - требуем ХОТЯ БЫ ОДНО
    # Баланс между качеством и количеством: требуем либо всплеск объема, либо momentum
    volume_or_momentum_ok = vol_spike or momentum_ok  # Требуем ХОТЯ БЫ ОДНО условие
    
    if not (trend_ok and rsi_ok and volume_or_momentum_ok):
        failed_checks = []
        if not trend_ok:
            failed_checks.append(f"trend (EMA12: {last_ema_fast:.6g}, EMA26: {last_ema_slow:.6g})")
        if not rsi_ok:
            failed_checks.append(f"RSI ({last_rsi:.1f}, требуется {config.RSI_LONG_MIN if side == 'LONG' else config.RSI_SHORT_MIN}-{config.RSI_LONG_MAX if side == 'LONG' else config.RSI_SHORT_MAX})")
        if not volume_or_momentum_ok:
            failed_checks.append(f"volume OR momentum (vol_spike: {vol_spike}, momentum: {momentum_ok}) - требуется ХОТЯ БЫ ОДНО")
        logging.info(f"{symbol} {side}: ❌ основные проверки не прошли: {', '.join(failed_checks)}")
        return None
    
    # Проверка MACD для подтверждения тренда (опциональна, влияет на score)
    # MACD теперь не блокирует сигнал, но дает бонусы в score
    macd_confirm = False
    if config.USE_MACD:
        if side == "LONG" and last_macd_hist > 0:
            macd_confirm = True
        elif side == "SHORT" and last_macd_hist < 0:
            macd_confirm = True
        # Не блокируем сигнал, если MACD не подтверждает - просто не даем бонусы
    
    # Проверка ADX для силы тренда (опциональна, влияет на score)
    # ADX теперь не блокирует сигнал, но дает бонусы в score
    adx_strong = False
    if config.USE_ADX:
        if last_adx >= config.MIN_ADX:
            adx_strong = True
        # Не блокируем сигнал, если ADX слабый - просто не даем бонусы
    
    # Дополнительный фильтр: проверка волатильности через ATR
    # Избегаем входов в периоды слишком высокой волатильности (риск больших проскальзываний)
    if last_atr > 0:
        atr_to_price_pct = (last_atr / last_close) * 100
        # Если ATR больше 3% от цены, волатильность слишком высока
        if atr_to_price_pct > 3.0:
            logging.info(f"{symbol} {side}: ❌ волатильность слишком высока (ATR: {atr_to_price_pct:.2f}% от цены)")
            return None

    if config.BTC_TREND_FILTER and market_trend in ("UP", "DOWN"):
        if side == "LONG" and market_trend == "DOWN":
            logging.info(f"{symbol} {side}: ❌ BTC тренд фильтр (BTC: {market_trend})")
            return None
        if side == "SHORT" and market_trend == "UP":
            logging.info(f"{symbol} {side}: ❌ BTC тренд фильтр (BTC: {market_trend})")
            return None
    
    # Вычисляем оценку качества сигнала
    price_change_24h = float(ticker_row["priceChangePercent"])
    signal_score = calculate_signal_score(
        side=side,
        rsi_val=last_rsi,
        ema_fast_val=last_ema_fast,
        ema_slow_val=last_ema_slow,
        macd_hist=last_macd_hist,
        adx_val=last_adx,
        vol_spike=vol_spike,
        momentum_ok=momentum_ok,
        recent_change_pct=recent_change_pct,
        price_change_24h=price_change_24h,
    )
    
    # Минимальная оценка для принятия сигнала (сбалансировано для получения нескольких сигналов в день)
    MIN_SCORE_THRESHOLD = 50.0  # Сбалансировано для получения нескольких сигналов в день
    if signal_score < MIN_SCORE_THRESHOLD:
        logging.info(f"{symbol} {side}: ❌ score слишком низкий ({signal_score:.1f} < {MIN_SCORE_THRESHOLD})")
        return None

    if side == "LONG":
        entry = last_close
        sl = entry - config.ATR_SL_MULTIPLIER * last_atr
        tp1 = entry + config.ATR_TP1_MULTIPLIER * last_atr
        tp2 = entry + config.ATR_TP2_MULTIPLIER * last_atr
    else:
        entry = last_close
        sl = entry + config.ATR_SL_MULTIPLIER * last_atr
        tp1 = entry - config.ATR_TP1_MULTIPLIER * last_atr
        tp2 = entry - config.ATR_TP2_MULTIPLIER * last_atr
    
    # Проверка минимального Risk/Reward соотношения перед входом
    # Это критично для прибыльности стратегии
    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    if risk > 0:
        risk_reward_ratio = reward / risk
        MIN_RISK_REWARD_RATIO = 1.8  # Минимум 1.8:1 для прибыльности
        if risk_reward_ratio < MIN_RISK_REWARD_RATIO:
            logging.info(f"{symbol} {side}: ❌ Risk/Reward слишком низкий ({risk_reward_ratio:.2f} < {MIN_RISK_REWARD_RATIO})")
            return None

    price_change = float(ticker_row["priceChangePercent"])
    high_price = float(ticker_row["highPrice"])
    low_price = float(ticker_row["lowPrice"])
    last_price = float(ticker_row["lastPrice"])

    tag_parts = []
    # Добавляем информацию о раннем входе
    if abs(recent_change_pct) >= config.MIN_RECENT_CHANGE_PCT:
        tag_parts.append(f"Early move ({recent_change_pct:+.1f}% recent)")
    
    if momentum_ok:
        tag_parts.append("Momentum")
    
    if side == "LONG" and price_change > 5:
        tag_parts.append("Rally")
    if side == "SHORT" and price_change < -5:
        tag_parts.append("Dump")
    if last_price > 0.98 * high_price:
        tag_parts.append("Near 24h High")
    if last_price < 1.02 * low_price:
        tag_parts.append("Near 24h Low")

    tag = ", ".join(tag_parts) if tag_parts else "Normal"

    # Формируем причину с учетом раннего обнаружения
    reason_parts = [f"Score: {signal_score:.1f}"]
    if momentum_ok:
        reason_parts.append("Momentum")
    if config.EMA_CROSS_RECENT:
        reason_parts.append("EMA cross")
    if config.RSI_ENTRY_CHECK:
        reason_parts.append("RSI entry")
    if vol_spike:
        reason_parts.append("Volume spike")
    if config.USE_MACD:
        reason_parts.append("MACD confirm")
    if config.USE_ADX:
        reason_parts.append(f"ADX {last_adx:.1f}")
    reason_parts.append(f"Trend ({config.TIMEFRAME_MAIN})")
    reason = " | ".join(reason_parts)
    
    return Signal(
        symbol=symbol,
        side=side,
        reason=reason,
        timeframe=config.TIMEFRAME_MAIN,
        trend_tf=config.TIMEFRAME_TREND,
        last_price=last_close,
        rsi=last_rsi,
        ema_fast=last_ema_fast,
        ema_slow=last_ema_slow,
        atr=last_atr,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        volume_24h=float(ticker_row["quoteVolume"]),
        change_24h=price_change,
        tag=tag,
        score=signal_score,
    )

