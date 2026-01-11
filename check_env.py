"""Скрипт для проверки настроек из .env"""
import os
import sys
from dotenv import load_dotenv

# Устанавливаем UTF-8 для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print("=" * 80)
print("Проверка настроек из .env")
print("=" * 80)

# Проверяем основные настройки
exchange = os.getenv("EXCHANGE", "bybit")
bybit_base = os.getenv("BYBIT_FUTURES_BASE", "https://api-testnet.bybit.com")
api_key = os.getenv("BYBIT_API_KEY", "")
api_secret = os.getenv("BYBIT_API_SECRET", "")

print(f"EXCHANGE: {exchange}")
print(f"BYBIT_FUTURES_BASE: {bybit_base}")
print(f"BYBIT_API_KEY: {api_key[:15]}... (длина: {len(api_key)})" if api_key else "BYBIT_API_KEY: НЕ ЗАДАН")
print(f"BYBIT_API_SECRET: {'*' * 15}... (длина: {len(api_secret)})" if api_secret else "BYBIT_API_SECRET: НЕ ЗАДАН")

print()
print("Важно:")
if "demo" in bybit_base.lower():
    print("  Используется DEMO API (демо-торговля)")
    print("  Ключи должны быть с https://www.bybit.com (demo режим)")
elif "testnet" in bybit_base.lower():
    print("  Используется TESTNET (демо-торговля)")
    print("  Ключи должны быть с https://testnet.bybit.com")
else:
    print("  Используется MAINNET (реальная торговля!)")
    print("  Ключи должны быть с https://www.bybit.com")

print()
if not api_key or not api_secret:
    print("ОШИБКА: API ключи не заданы!")
elif len(api_key) < 20:
    print(f"ВНИМАНИЕ: API ключ слишком короткий ({len(api_key)} символов)")
    print("  Обычно API ключи ByBit длиннее. Проверьте, что скопировали полностью.")
elif len(api_secret) < 20:
    print(f"ВНИМАНИЕ: API Secret слишком короткий ({len(api_secret)} символов)")
    print("  Обычно API Secret длиннее. Проверьте, что скопировали полностью.")
else:
    print("OK: API ключи заданы (проверьте правильность через test_api_keys.py)")

print()
print("=" * 80)

