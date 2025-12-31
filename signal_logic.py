"""Модуль логики генерации торговых сигналов"""
import math
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional

import config
import binance_api
import indicators
from models import Signal


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
        df = binance_api.get_klines("BTCUSDT", config.TIMEFRAME_TREND, limit=200)
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
    
    # RSI в оптимальной зоне (20 баллов)
    if side == "LONG":
        if 50 <= rsi_val <= 60:  # Идеальная зона для входа в LONG
            score += 20
        elif 45 <= rsi_val < 50 or 60 < rsi_val <= 65:
            score += 10
    else:  # SHORT
        if 40 <= rsi_val <= 50:  # Идеальная зона для входа в SHORT
            score += 20
        elif 35 <= rsi_val < 40 or 50 < rsi_val <= 55:
            score += 10
    
    # EMA пересечение и сила тренда (25 баллов)
    if side == "LONG":
        if ema_fast_val > ema_slow_val:
            ema_diff_pct = ((ema_fast_val - ema_slow_val) / ema_slow_val) * 100
            score += min(25, ema_diff_pct * 2)  # До 25 баллов за сильный тренд
    else:  # SHORT
        if ema_fast_val < ema_slow_val:
            ema_diff_pct = ((ema_slow_val - ema_fast_val) / ema_slow_val) * 100
            score += min(25, ema_diff_pct * 2)  # До 25 баллов за сильный тренд
    
    # MACD подтверждение (15 баллов)
    if config.USE_MACD:
        if side == "LONG" and macd_hist > 0:
            score += 15
        elif side == "SHORT" and macd_hist < 0:
            score += 15
    
    # ADX сила тренда (15 баллов)
    if config.USE_ADX:
        if adx_val >= config.MIN_ADX:
            score += min(15, (adx_val - config.MIN_ADX) / 2)  # До 15 баллов за сильный тренд
    
    # Объем (10 баллов)
    if vol_spike:
        score += 10
    
    # Momentum (10 баллов)
    if momentum_ok:
        score += 10
    
    # Недавнее движение (5 баллов)
    if abs(recent_change_pct) >= config.MIN_RECENT_CHANGE_PCT:
        score += 5
    
    return min(100.0, score)


def build_signal(symbol: str, side: str, ticker_row: Dict, market_trend: str) -> Optional[Signal]:
    """Строит торговый сигнал для символа"""
    try:
        df_main = binance_api.get_klines(symbol, config.TIMEFRAME_MAIN, limit=200)
        _ = binance_api.get_klines(symbol, config.TIMEFRAME_TREND, limit=200)
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
    
    # Улучшенная проверка momentum (ускорение цены) для раннего обнаружения
    momentum_ok = False
    if len(close) >= 3:
        # Проверяем ускорение: цена должна расти/падать быстрее
        price_change_1 = (last_close - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        price_change_2 = (float(close.iloc[-2]) - float(close.iloc[-3])) / float(close.iloc[-3]) * 100
        
        if side == "LONG":
            # Для LONG: требуется ускорение роста (более строго)
            momentum_ok = price_change_1 > price_change_2 and price_change_1 > 0.15  # Ускорение и минимум 0.15%
        else:
            # Для SHORT: требуется ускорение падения (более строго)
            momentum_ok = price_change_1 < price_change_2 and price_change_1 < -0.15  # Ускорение и минимум 0.15%

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
            return None
        
        if side == "LONG":
            recent_move_ok = recent_change_pct >= config.MIN_RECENT_CHANGE_PCT
        else:
            recent_move_ok = recent_change_pct <= -config.MIN_RECENT_CHANGE_PCT
        
        if not recent_move_ok:
            return None
    
    # 2. Проверка, что RSI только что вошел в нужную зону - ОПЦИОНАЛЬНО
    rsi_entry_ok = True
    if config.RSI_ENTRY_CHECK:
        if len(rsi_series) < 3:
            return None
        
        prev_rsi = float(rsi_series.iloc[-2])
        prev_prev_rsi = float(rsi_series.iloc[-3])
        
        if side == "LONG":
            # RSI должен был быть ниже зоны и только что войти в нее, ИЛИ находиться в начале зоны (первые 50%)
            rsi_just_entered = (prev_rsi < config.RSI_LONG_MIN or prev_prev_rsi < config.RSI_LONG_MIN) and (config.RSI_LONG_MIN <= last_rsi <= config.RSI_LONG_MAX)
            rsi_in_early_zone = config.RSI_LONG_MIN <= last_rsi <= (config.RSI_LONG_MIN + (config.RSI_LONG_MAX - config.RSI_LONG_MIN) * 0.5)  # Первые 50% зоны
            rsi_rising = last_rsi > prev_rsi  # RSI растет
            rsi_entry_ok = (rsi_just_entered or (rsi_in_early_zone and rsi_rising))  # Более строго
        else:
            # RSI должен был быть выше зоны и только что войти в нее, ИЛИ находиться в начале зоны (первые 50%)
            rsi_just_entered = (prev_rsi > config.RSI_SHORT_MAX or prev_prev_rsi > config.RSI_SHORT_MAX) and (config.RSI_SHORT_MIN <= last_rsi <= config.RSI_SHORT_MAX)
            rsi_in_early_zone = (config.RSI_SHORT_MIN + (config.RSI_SHORT_MAX - config.RSI_SHORT_MIN) * 0.5) <= last_rsi <= config.RSI_SHORT_MAX  # Последние 50% зоны
            rsi_falling = last_rsi < prev_rsi  # RSI падает
            rsi_entry_ok = (rsi_just_entered or (rsi_in_early_zone and rsi_falling))  # Более строго
        
        if not rsi_entry_ok:
            return None
    
    # 3. Проверка недавнего пересечения EMA - ОПЦИОНАЛЬНО
    ema_cross_ok = True
    if config.EMA_CROSS_RECENT:
        if len(ema_fast) < 3 or len(ema_slow) < 3:
            return None
        
        prev_ema_fast = float(ema_fast.iloc[-2])
        prev_ema_slow = float(ema_slow.iloc[-2])
        prev_prev_ema_fast = float(ema_fast.iloc[-3]) if len(ema_fast) >= 3 else prev_ema_fast
        prev_prev_ema_slow = float(ema_slow.iloc[-3]) if len(ema_slow) >= 3 else prev_ema_slow
        
        if side == "LONG":
            # EMA должны были пересечься недавно (более строго)
            ema_crossed = (prev_ema_fast <= prev_ema_slow or prev_prev_ema_fast <= prev_prev_ema_slow) and (last_ema_fast > last_ema_slow)
            ema_converging = (last_ema_fast - last_ema_slow) > (prev_ema_fast - prev_ema_slow)  # Сближаются
            ema_cross_ok = ema_crossed or (ema_converging and last_ema_fast > last_ema_slow * 0.998)  # Почти пересекли
        else:
            # Для SHORT: аналогично
            ema_crossed = (prev_ema_fast >= prev_ema_slow or prev_prev_ema_fast >= prev_prev_ema_slow) and (last_ema_fast < last_ema_slow)
            ema_converging = (last_ema_slow - last_ema_fast) > (prev_ema_slow - prev_ema_fast)  # Сближаются
            ema_cross_ok = ema_crossed or (ema_converging and last_ema_fast < last_ema_slow * 1.002)  # Почти пересекли
        
        if not ema_cross_ok:
            return None
    
    # 4. Проверка, что объем начал расти недавно - ОПЦИОНАЛЬНО
    vol_recent_ok = True
    if config.VOL_RECENT_CHECK:
        if len(vol) >= config.RECENT_CANDLES_LOOKBACK:
            recent_vols = vol.iloc[-config.RECENT_CANDLES_LOOKBACK:].astype(float)
            recent_avg_vol = float(recent_vols.mean())
            # Объем за последние N свечей должен быть выше среднего (ужесточено до 5%)
            vol_recent_ok = recent_avg_vol > avg_vol * 1.05  # 5% выше среднего для качества
        
        if not vol_recent_ok:
            return None
    
    # ========== ОСНОВНЫЕ ПРОВЕРКИ ==========
    
    trend_ok = False
    rsi_ok = False

    if side == "LONG":
        # Для раннего обнаружения: EMA быстрая должна быть близка к медленной или выше
        ema_fast_prev = float(ema_fast.iloc[-2]) if len(ema_fast) >= 2 else last_ema_fast
        ema_slow_prev = float(ema_slow.iloc[-2]) if len(ema_slow) >= 2 else last_ema_slow
        ema_converging = (last_ema_fast > last_ema_slow) or \
                        ((last_ema_fast - last_ema_slow) > (ema_fast_prev - ema_slow_prev))  # Сближаются
        trend_ok = ema_converging or last_ema_fast > last_ema_slow
        rsi_ok = config.RSI_LONG_MIN <= last_rsi <= config.RSI_LONG_MAX
    else:
        # Для SHORT: аналогично
        ema_fast_prev = float(ema_fast.iloc[-2]) if len(ema_fast) >= 2 else last_ema_fast
        ema_slow_prev = float(ema_slow.iloc[-2]) if len(ema_slow) >= 2 else last_ema_slow
        ema_converging = (last_ema_fast < last_ema_slow) or \
                        ((last_ema_slow - last_ema_fast) > (ema_slow_prev - ema_fast_prev))  # Сближаются
        trend_ok = ema_converging or last_ema_fast < last_ema_slow
        rsi_ok = config.RSI_SHORT_MIN <= last_rsi <= config.RSI_SHORT_MAX

    # Основные проверки: тренд, RSI, и (всплеск объема ИЛИ momentum)
    volume_or_momentum_ok = vol_spike or momentum_ok
    
    if not (trend_ok and rsi_ok and volume_or_momentum_ok):
        return None
    
    # Проверка MACD для подтверждения тренда
    if config.USE_MACD:
        if side == "LONG" and last_macd_hist <= 0:
            return None
        if side == "SHORT" and last_macd_hist >= 0:
            return None
    
    # Проверка ADX для силы тренда
    if config.USE_ADX:
        if last_adx < config.MIN_ADX:
            return None

    if config.BTC_TREND_FILTER and market_trend in ("UP", "DOWN"):
        if side == "LONG" and market_trend == "DOWN":
            return None
        if side == "SHORT" and market_trend == "UP":
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
    
    # Минимальная оценка для принятия сигнала (ужесточено)
    if signal_score < 40.0:  # Минимум 40 баллов для принятия сигнала
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

