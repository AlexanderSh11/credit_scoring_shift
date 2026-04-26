import os

DB_USER = os.getenv("DB_USER")

DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_FULL_PATH = "C:/csv_files"

DATABASE_NAME = "home_credit"

DB_ARGS = {
    "database": DATABASE_NAME,
    "host": "127.0.0.1",
    "port": "5433",
    "user": DB_USER,
    "password": DB_PASSWORD,
}

class CalculatorConfig:
    MAX_AMOUNT = 300_000
    HIGH_AMOUNT = 200_000
    MEDIUM_AMOUNT = 50_000
    MIN_AMOUNT = 20_000
    DAYS_IN_YEAR = 365.2425
