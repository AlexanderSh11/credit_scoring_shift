import argparse
import json
import re
import pandas as pd
from tqdm import tqdm
import time


def load_logs(log_file_path, n_lines=10):
    """
    Загружает логи из файла.

    :param log_file_path: путь до файла с логами
    :param n_lines: количество строк логов для парсинга (None - для загрузки всего файла)
    :return data: список строк из файла
    """
    data = []
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            with tqdm(desc="Загрузка логов", unit="строк") as pbar:
                for i, line in enumerate(f):
                    if n_lines is not None and i >= n_lines:
                        break

                    logs_obj = json.loads(line)
                    data.append(logs_obj)
                    pbar.update(1)
        return data
    except FileNotFoundError:
        print(f"Ошибка: файл {log_file_path} не найден")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return []


def parse_logs(data):
    """
    Парсит логи.

    :param data: список строк логов
    :return bureau_parsed_logs: список записей для файла bureau.csv
    :return POS_CASH_balance_parsed_logs: список записей для файла POS_CASH_balance.csv
    """
    bureau_parsed_logs = []
    POS_CASH_balance_parsed_logs = []
    with tqdm(total=len(data), desc="Парсинг", unit="записей") as pbar:
        for logs in data:
            if logs["type"] == "bureau":
                parsed_bureau = parse_bureau_log(logs["data"])
                bureau_parsed_logs.append(parsed_bureau)
            elif logs["type"] == "POS_CASH_balance":
                parsed_POS_CASH_balance = parse_POS_CASH_balance_log(logs["data"])
                # extend добавляет элементы списка по одному, а не список целиком
                POS_CASH_balance_parsed_logs.extend(parsed_POS_CASH_balance)

            pbar.update(1)

    return bureau_parsed_logs, POS_CASH_balance_parsed_logs


def parse_amt_credit_string(amt_credit_str):
    """
    Парсит строку вида "AmtCredit(CREDIT_CURRENCY='currency 1', ...)" и возвращает словарь с полями.

    :param amt_credit_str: экземпляр объекта AmtCredit в виде строки
    :return dict: словарь с полями
    """
    # (\w+) - имя параметра (буквы/цифры)
    # =([^,)]+) - значение до запятой или закрывающей скобки
    # [^,)]+ - один или более символов, кроме запятой и скобки
    pattern = r"(\w+)=([^,)]+)"
    matches = re.findall(pattern, amt_credit_str)

    params = {}
    for key, value in matches:
        value = value.strip()

        if value == "None":
            params[key] = None
        # Строки в кавычках: удаляем кавычки, оставляем содержимое
        elif value.startswith("'") and value.endswith("'"):
            params[key] = value[1:-1]

        else:
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    return {
        "CREDIT_CURRENCY": params.get("CREDIT_CURRENCY"),
        "AMT_CREDIT_MAX_OVERDUE": params.get("AMT_CREDIT_MAX_OVERDUE"),
        "AMT_CREDIT_SUM": params.get("AMT_CREDIT_SUM"),
        "AMT_CREDIT_SUM_DEBT": params.get("AMT_CREDIT_SUM_DEBT"),
        "AMT_CREDIT_SUM_LIMIT": params.get("AMT_CREDIT_SUM_LIMIT"),
        "AMT_CREDIT_SUM_OVERDUE": params.get("AMT_CREDIT_SUM_OVERDUE"),
        "AMT_ANNUITY": params.get("AMT_ANNUITY"),
    }


def parse_bureau_log(logs):
    """
    Парсит bureau запись и возвращает плоский словарь.

    :param logs: словарь с данными bureau из JSON
    :return parsed_log: словарь с полями для CSV
    """
    parsed_log = {}
    parsed_log["CREDIT_TYPE"] = logs["CREDIT_TYPE"]
    record = logs.get("record", {})
    parsed_log["SK_ID_CURR"] = record.get("SK_ID_CURR")
    parsed_log["SK_ID_BUREAU"] = record.get("SK_ID_BUREAU")
    parsed_log["CREDIT_ACTIVE"] = record.get("CREDIT_ACTIVE")
    parsed_log["DAYS_CREDIT"] = record.get("DAYS_CREDIT")
    parsed_log["CREDIT_DAY_OVERDUE"] = record.get("CREDIT_DAY_OVERDUE")
    parsed_log["DAYS_CREDIT_ENDDATE"] = record.get("DAYS_CREDIT_ENDDATE")
    parsed_log["DAYS_ENDDATE_FACT"] = record.get("DAYS_ENDDATE_FACT")
    parsed_log["CNT_CREDIT_PROLONG"] = record.get("CNT_CREDIT_PROLONG")
    parsed_log["DAYS_CREDIT_UPDATE"] = record.get("DAYS_CREDIT_UPDATE")
    amt_credit_str = record.get("AmtCredit", "")
    if amt_credit_str and isinstance(amt_credit_str, str):
        amt_credit_dict = parse_amt_credit_string(amt_credit_str)
    else:
        amt_credit_dict = {
            "CREDIT_CURRENCY": None,
            "AMT_CREDIT_MAX_OVERDUE": None,
            "AMT_CREDIT_SUM": None,
            "AMT_CREDIT_SUM_DEBT": None,
            "AMT_CREDIT_SUM_LIMIT": None,
            "AMT_CREDIT_SUM_OVERDUE": None,
            "AMT_ANNUITY": None,
        }

    parsed_log.update(amt_credit_dict)

    return parsed_log


def parse_POS_CASH_balance_log(logs):
    """
    Парсит POS_CASH_balance запись и возвращает список плоских словарей.

    :param logs: словарь с данными POS_CASH_balance из JSON
    :return parsed_logs: список словарей с полями для CSV
    """
    parsed_logs = []
    # CNT_INSTALMENT повторяется для каждой вложенной записи
    cnt_instalment = logs.get("CNT_INSTALMENT")

    for record in logs.get("records", []):
        parsed_record = {}

        parsed_record["CNT_INSTALMENT"] = cnt_instalment
        parsed_record["CNT_INSTALMENT_FUTURE"] = record.get("CNT_INSTALMENT_FUTURE")
        parsed_record["MONTHS_BALANCE"] = record.get("MONTHS_BALANCE")
        parsed_record["SK_DPD"] = record.get("SK_DPD")
        parsed_record["SK_DPD_DEF"] = record.get("SK_DPD_DEF")
        # PosCashBalanceIDs в логе хранится как строка, а не как JSON объект
        pos_cash_ids = record.get("PosCashBalanceIDs")
        # re.search находит первое вхождение паттерна в строке
        # \d+ - одна или более цифр
        sk_id_prev_match = re.search(r"SK_ID_PREV=(\d+)", pos_cash_ids)
        sk_id_curr_match = re.search(r"SK_ID_CURR=(\d+)", pos_cash_ids)
        # [^']+ - один или более символов, кроме кавычки
        name_match = re.search(r"NAME_CONTRACT_STATUS='([^']+)'", pos_cash_ids)

        parsed_record["SK_ID_PREV"] = (
            # .group(1) берет первую захваченную группу
            int(sk_id_prev_match.group(1)) if sk_id_prev_match else None
        )
        parsed_record["SK_ID_CURR"] = (
            int(sk_id_curr_match.group(1)) if sk_id_curr_match else None
        )
        parsed_record["NAME_CONTRACT_STATUS"] = (
            name_match.group(1) if name_match else None
        )

        parsed_logs.append(parsed_record)

    return parsed_logs


def save_data_to_csv(data, path):
    """
    Сохраняет данные в CSV файл.

    :param data: список словарей для сохранения
    :param path: путь для сохранения CSV файла
    :return df: DataFrame с сохранёнными данными
    """
    if not data:
        print(f"Нет данных для сохранения: {path}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def parse_arguments():
    """
    Парсит аргументы командной строки.

    :return: объект с аргументами
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-file",
        type=str,
        default="POS_CASH_balance_plus_bureau-001-001.log",
        help="Путь к входному .log файлу",
    )

    parser.add_argument(
        "--bureau-csv",
        type=str,
        default="bureau_parsed.csv",
        help="Путь к выходному bureau.csv файлу",
    )

    parser.add_argument(
        "--pos-cash-csv",
        type=str,
        default="POS_CASH_balance_parsed.csv",
        help="Путь к выходному POS_CASH_balance.csv файлу",
    )

    return parser.parse_args()


def main():
    start_time = time.time()

    args = parse_arguments()

    log_file_path = args.log_file
    POS_CASH_balance_csv_file_path = args.pos_cash_csv
    bureau_csv_file_path = args.bureau_csv

    data = load_logs(log_file_path, n_lines=None)
    data_bureau, data_POS_CASH_balance = parse_logs(data)
    save_data_to_csv(data=data_bureau, path=bureau_csv_file_path)
    save_data_to_csv(data=data_POS_CASH_balance, path=POS_CASH_balance_csv_file_path)

    total_time = time.time() - start_time
    print(f"Время выполнения: {total_time:.2f} секунд")


if __name__ == "__main__":
    main()
