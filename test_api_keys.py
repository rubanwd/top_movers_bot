"""Скрипт для проверки API ключей ByBit"""
import os
import sys
from dotenv import load_dotenv
from bybit_api_new import BybitAPI
import logging

# Устанавливаем UTF-8 для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

base_url = os.getenv("BYBIT_FUTURES_BASE", "https://api-testnet.bybit.com")
api_key = os.getenv("BYBIT_API_KEY", "")
api_secret = os.getenv("BYBIT_API_SECRET", "")

print("=" * 80)
print("Проверка API ключей ByBit")
print("=" * 80)
print(f"Base URL: {base_url}")
print(f"API Key: {api_key[:10]}... (первые 10 символов)" if api_key else "API Key: НЕ ЗАДАН")
print(f"API Secret: {'*' * 10}... (скрыт)" if api_secret else "API Secret: НЕ ЗАДАН")
print()

if not api_key or not api_secret:
    print("ОШИБКА: API ключи не заданы в .env файле!")
    print("   Добавьте в .env:")
    print("   BYBIT_API_KEY=ваш_api_key")
    print("   BYBIT_API_SECRET=ваш_api_secret")
    exit(1)

api = BybitAPI(base_url=base_url, api_key=api_key, api_secret=api_secret)

print("Тест 1: Получение информации об аккаунте...")
try:
    account_info = api.get_account_info()
    result = account_info.get("result", {})
    balance_list = result.get("list", [])
    
    if balance_list:
        print("OK: Успешно получена информация об аккаунте!")
        for account in balance_list:
            coins = account.get("coin", [])
            for coin in coins:
                if coin.get("coin") == "USDT":
                    balance = float(coin.get("walletBalance", 0))
                    print(f"   Баланс USDT: {balance:.2f}")
    else:
        print("ВНИМАНИЕ: Аккаунт найден, но баланс пустой")
        
except Exception as e:
    print(f"ОШИБКА: {e}")
    print()
    print("Возможные причины:")
    print("1. API ключи неправильные или истекли")
    print("2. API ключи для другой среды (testnet/mainnet)")
    print("3. У API ключа нет прав на чтение баланса")
    print("4. Base URL не соответствует ключам")
    print()
    print("Проверьте:")
    print(f"  - Base URL: {base_url}")
    if "testnet" in base_url.lower():
        print("  - Убедитесь, что используете ключи с https://testnet.bybit.com")
    else:
        print("  - Убедитесь, что используете ключи с https://www.bybit.com")
    exit(1)

print()
print("Тест 2: Получение информации о символе BTCUSDT...")
try:
    symbol_info = api.get_symbol_info("BTCUSDT")
    result = symbol_info.get("result", {})
    instruments = result.get("list", [])
    if instruments:
        print("OK: Успешно получена информация о символе!")
        instrument = instruments[0]
        print(f"   Символ: {instrument.get('symbol')}")
        print(f"   Статус: {instrument.get('status')}")
    else:
        print("ВНИМАНИЕ: Символ не найден")
except Exception as e:
    print(f"ОШИБКА: {e}")

print()
print("=" * 80)
print("OK: Проверка завершена!")
print("=" * 80)

