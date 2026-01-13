# Руководство по оптимизации .env файла

## Важно!

Я изменил **значения по умолчанию** в `config.py`. Это означает:

- ✅ **Если в `.env` НЕТ этих переменных** - будут использоваться новые оптимизированные значения автоматически
- ⚠️ **Если в `.env` УЖЕ ЕСТЬ эти переменные** - они имеют приоритет и переопределят новые значения

## Что нужно проверить/изменить в .env

### 1. Параметры SL/TP (КРИТИЧНО для улучшения Risk/Reward)

Если у вас есть эти строки в `.env`, обновите их:

```env
# Старые значения (удалите или замените):
# ATR_SL_MULTIPLIER=1.5
# ATR_TP1_MULTIPLIER=2.0
# ATR_TP2_MULTIPLIER=3.0

# Новые оптимизированные значения:
ATR_SL_MULTIPLIER=1.2
ATR_TP1_MULTIPLIER=2.5
ATR_TP2_MULTIPLIER=4.0
```

### 2. Параметры RSI (для лучшего win rate)

Если у вас есть эти строки в `.env`, обновите их:

```env
# Старые значения:
# RSI_LONG_MIN=50
# RSI_LONG_MAX=60
# RSI_SHORT_MIN=40
# RSI_SHORT_MAX=50

# Новые оптимизированные значения:
RSI_LONG_MIN=48
RSI_LONG_MAX=58
RSI_SHORT_MIN=42
RSI_SHORT_MAX=52
```

### 3. Параметры объема и фильтров

Если у вас есть эти строки в `.env`, обновите их:

```env
# Старые значения:
# VOL_SPIKE_MULTIPLIER=2.0
# MIN_ADX=30.0
# MIN_RECENT_CHANGE_PCT=0.5
# MAX_SIGNALS_PER_DAY=3

# Новые оптимизированные значения:
VOL_SPIKE_MULTIPLIER=1.8
MIN_ADX=25.0
MIN_RECENT_CHANGE_PCT=0.4
MAX_SIGNALS_PER_DAY=5
```

## Рекомендация

### Вариант 1: Удалить переменные из .env (РЕКОМЕНДУЕТСЯ)
Если вы хотите использовать новые оптимизированные значения по умолчанию, просто **удалите** соответствующие строки из `.env`. Тогда будут использоваться новые значения из `config.py`.

### Вариант 2: Обновить значения в .env
Если вы хотите явно указать значения в `.env`, обновите их согласно списку выше.

## Как проверить текущие значения

Запустите бота и посмотрите в логах - там будут выведены все загруженные настройки:
```
Загружены настройки: TOP_N=8, SCAN_INTERVAL_SECONDS=600, MIN_QUOTE_VOLUME_USDT=5000000
```

Также в логах при старте будут показаны значения RSI, ATR множителей и других параметров.

## Важные переменные, которые НЕ нужно менять

Эти переменные остаются без изменений:
- `EXCHANGE`
- `BYBIT_FUTURES_BASE`
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `SCAN_INTERVAL_SECONDS`
- `TOP_N`
- `MIN_QUOTE_VOLUME_USDT`
- `TIMEFRAME_MAIN`
- `TIMEFRAME_TREND`

## Пример минимального .env с оптимизированными значениями

```env
# Биржа
EXCHANGE=bybit
BYBIT_FUTURES_BASE=https://api.bybit.com
BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here
BYBIT_ENABLE_TRADING=1
BYBIT_RISK_PERCENT=1.0

# Telegram
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Основные настройки
SCAN_INTERVAL_SECONDS=600
TOP_N=8
MIN_QUOTE_VOLUME_USDT=5000000

# Таймфреймы
TIMEFRAME_MAIN=5m
TIMEFRAME_TREND=1h

# ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ (можно добавить для явного указания):
# ATR_SL_MULTIPLIER=1.2
# ATR_TP1_MULTIPLIER=2.5
# ATR_TP2_MULTIPLIER=4.0
# RSI_LONG_MIN=48
# RSI_LONG_MAX=58
# RSI_SHORT_MIN=42
# RSI_SHORT_MAX=52
# VOL_SPIKE_MULTIPLIER=1.8
# MIN_ADX=25.0
# MIN_RECENT_CHANGE_PCT=0.4
# MAX_SIGNALS_PER_DAY=5
```

## Итог

**Короткий ответ:** Если в вашем `.env` нет переменных из списка выше - ничего менять не нужно! Новые оптимизированные значения будут использоваться автоматически.

Если они есть - либо удалите их, либо обновите на новые значения.
