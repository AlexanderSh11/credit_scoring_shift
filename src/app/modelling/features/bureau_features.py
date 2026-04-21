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


def generate_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков из bureau таблицы
    """

    # Копируем индекс
    features = df[["sk_id_curr"]].copy().drop_duplicates().reset_index(drop=True)

    # 1. Максимальная сумма просрочки (если просрочек нет, то NaN)
    overdue_df = df[df["amt_credit_sum_overdue"] > 0]
    max_overdue = overdue_df.groupby("sk_id_curr")["amt_credit_sum_overdue"].max()
    features["max_overdue"] = features["sk_id_curr"].map(max_overdue)

    # 2. Минимальная сумма просрочки (если просрочек нет, то NaN, зачем считать непросроченный кредит минимальной просрочкой?)
    min_overdue = overdue_df.groupby("sk_id_curr")["amt_credit_sum_overdue"].min()
    features["min_overdue"] = features["sk_id_curr"].map(min_overdue)

    # 3. Какую долю суммы от открытого займа просрочил
    open_df = df[df["credit_active"] == "Active"]
    total_credit = open_df.groupby("sk_id_curr")["amt_credit_sum"].sum()
    total_overdue = open_df.groupby("sk_id_curr")["amt_credit_sum_overdue"].sum()
    features["overdue_proportion"] = features["sk_id_curr"].map(
        total_overdue
    ) / features["sk_id_curr"].map(total_credit)
    features["overdue_proportion"] = features["overdue_proportion"].replace(
        [np.inf, -np.inf], np.nan
    )

    # 4. Кол-во кредитов определенного типа
    # Группируем данные по клиенту (sk_id_curr) и типу кредита (credit_type)
    # fill_value=0 заполняет нулями отсутствующие комбинации
    credit_type_counts = (
        df.groupby(["sk_id_curr", "credit_type"]).size().unstack(fill_value=0)
    )
    # Переименовываем колонки
    credit_type_counts.columns = [
        f"credit_type_{col}" for col in credit_type_counts.columns
    ]
    # Объединяем с features, how="left" сохраняет всех клиентов, даже если у них нет кредитов в bureau
    features = features.merge(credit_type_counts, on="sk_id_curr", how="left")

    # 5. Кол-во просрочек кредитов определенного типа
    overdue_df = df[df["amt_credit_sum_overdue"] > 0]
    overdue_counts = (
        overdue_df.groupby(["sk_id_curr", "credit_type"]).size().unstack(fill_value=0)
    )
    overdue_counts.columns = [f"overdue_type_{col}" for col in overdue_counts.columns]
    features = features.merge(overdue_counts, on="sk_id_curr", how="left")

    # 6. Кол-во закрытых кредитов определенного типа
    closed_df = df[df["credit_active"] == "Closed"]
    closed_counts = (
        closed_df.groupby(["sk_id_curr", "credit_type"]).size().unstack(fill_value=0)
    )
    closed_counts.columns = [f"closed_type_{col}" for col in closed_counts.columns]
    features = features.merge(closed_counts, on="sk_id_curr", how="left")
    return features


def main():
    """Загрузка данных и генерация признаков для bureau"""

    db_manager = DatabaseManager()
    table_name = "bureau"
    query = f"""
    SELECT * FROM {table_name}
    """
    df = db_manager.get_df_from_query(query)

    bureau_features = generate_bureau_features(df)
    current_dir = os.path.dirname(current_file)
    output_path = os.path.join(current_dir, "data/bureau_features.csv")
    bureau_features.to_csv(output_path, index=False)

    print(bureau_features.head(20))


if __name__ == "__main__":
    main()
