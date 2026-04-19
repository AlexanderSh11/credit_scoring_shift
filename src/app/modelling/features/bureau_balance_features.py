import os
import sys
import numpy as np
import pandas as pd

# Получаем абсолютный путь к текущему файлу
current_file = os.path.abspath(__file__)
# Переходим в credit_scoring
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
)
sys.path.insert(0, project_root)

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def generate_bureau_balance_features(balance_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков из bureau_balance таблицы
    """

    # Соединяем bureau_balance с bureau для получения sk_id_curr
    merged_df = balance_df.merge(
        bureau_df[["sk_id_bureau", "sk_id_curr"]], on="sk_id_bureau", how="inner"
    )
    features = merged_df[["sk_id_curr"]].copy().drop_duplicates().reset_index(drop=True)

    # Найдем последний статус по каждому кредиту
    latest_status = (
        merged_df.sort_values("months_balance").groupby("sk_id_bureau").last()
    )
    latest_status = latest_status.reset_index()[
        ["sk_id_bureau", "sk_id_curr", "status"]
    ]
    merged_df = merged_df.merge(
        latest_status, on="sk_id_bureau", suffixes=("", "_latest")
    )

    # 1. Кол-во открытых кредитов
    # C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,… 5 means DPD 120+ or sold or written off
    open_df = latest_status[~latest_status["status"].isin(["C", "X"])]
    open_counts = open_df.groupby("sk_id_curr")["sk_id_bureau"].count()
    features["open_credits_count"] = features["sk_id_curr"].map(open_counts).fillna(0)

    # 2. Кол-во закрытых кредитов
    closed_df = latest_status[latest_status["status"] == "C"]
    closed_counts = closed_df.groupby("sk_id_curr")["sk_id_bureau"].count()
    features["closed_credits_count"] = (
        features["sk_id_curr"].map(closed_counts).fillna(0)
    )

    # 3. Кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    overdue_df = latest_status[~latest_status["status"].isin(["C", "X", "0"])]
    overdue_counts = (
        overdue_df.groupby(["sk_id_curr", "status"])["sk_id_bureau"]
        .nunique()
        .unstack(fill_value=0)
    )
    overdue_counts.columns = [f"overdue_count_{col}" for col in overdue_counts.columns]
    features = features.merge(overdue_counts, on="sk_id_curr", how="left").fillna(0)

    # 4. Кол-во кредитов
    total_counts = latest_status.groupby("sk_id_curr")["sk_id_bureau"].count()
    features["total_credits_count"] = features["sk_id_curr"].map(total_counts).fillna(0)

    # 5. Доля закрытых кредитов (если кредитов нет, то 0)
    features["closed_to_total_credits_proportion"] = (
        features["closed_credits_count"] / features["total_credits_count"]
    )
    features["closed_to_total_credits_proportion"] = features[
        "closed_to_total_credits_proportion"
    ].replace([np.inf, -np.inf], 0)

    # 6. Доля открытых кредитов (если кредитов нет, то 0)
    features["open_to_total_credits_proportion"] = (
        features["open_credits_count"] / features["total_credits_count"]
    )
    features["open_to_total_credits_proportion"] = features[
        "open_to_total_credits_proportion"
    ].replace([np.inf, -np.inf], 0)

    # 7. Доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS) (если кредитов нет, то 0)
    for col in overdue_counts.columns:
        col_name = f"{col}_to_total_credits_proportion"
        features[col_name] = features[col] / features["total_credits_count"]
        features[col_name] = features[col_name].replace([np.inf, -np.inf], 0)

    # 8. Интервал между последним закрытым кредитом и текущей заявкой
    # Интервал = 0 - (минимальный months_balance среди статусов "C" для кредита)
    # Нам нужен минимальный такой интервал среди всех кредитов клиента, т.к. нужен последний закрытый кредит
    # Для каждого кредита находим минимальный months_balance где status = "C"
    closed_credits = merged_df[merged_df["status_latest"] == "C"]
    # Для клиента берем максимальный months_balance (самый свежий закрытый кредит)
    last_closed_per_client = closed_credits.groupby("sk_id_curr")[
        "months_balance"
    ].max()
    features["last_closed_interval"] = (
        features["sk_id_curr"].map(last_closed_per_client).fillna(0)
    )
    # интервал должен быть положительным
    features["last_closed_interval"] = -features["last_closed_interval"]

    # 9. Интервал между взятием последнего активного займа и текущей заявкой
    # Активные кредиты: все статусы кроме "C" (закрыт) и "X" (неизвестно)
    active_credits = merged_df[~merged_df["status_latest"].isin(["C", "X"])]
    # Для каждого кредита берем минимальный months_balance (момент взятия кредита)
    active_month_per_credit = active_credits.groupby("sk_id_bureau")[
        "months_balance"
    ].min()
    active_with_curr = active_month_per_credit.reset_index().merge(
        bureau_df[["sk_id_bureau", "sk_id_curr"]], on="sk_id_bureau"
    )
    last_active_per_client = active_with_curr.groupby("sk_id_curr")[
        "months_balance"
    ].max()
    features["last_active_interval"] = (
        features["sk_id_curr"].map(last_active_per_client).fillna(0)
    )
    features["last_active_interval"] = -features["last_active_interval"]

    return features


def main():
    """Загрузка данных и генерация признаков для bureau_balance"""

    db_manager = DatabaseManager()

    bureau_table_name = "bureau"
    query = f"""
    SELECT * FROM {bureau_table_name}
    """
    bureau_df = db_manager.get_df_from_query(query)
    balance_table_name = "bureau_balance"
    query = f"""
    SELECT * FROM {balance_table_name}
    """
    balance_df = db_manager.get_df_from_query(query)

    bureau_balance_features = generate_bureau_balance_features(
        balance_df=balance_df, bureau_df=bureau_df
    )
    current_dir = os.path.dirname(current_file)
    output_path = os.path.join(current_dir, "data/bureau_balance_features.csv")
    bureau_balance_features.to_csv(output_path, index=False)

    print(bureau_balance_features.head(20))


if __name__ == "__main__":
    main()
