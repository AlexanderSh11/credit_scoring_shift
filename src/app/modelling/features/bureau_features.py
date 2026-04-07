import os
import sys

# Получаем абсолютный путь к текущему файлу
current_file = os.path.abspath(__file__)
# Переходим в credit_scoring
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
)
sys.path.insert(0, project_root)

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def generate_bureau_features(df):
    """
    Генерация признаков из bureau таблицы
    """

    # Копируем индекс
    features = df[["sk_id_curr"]].copy()

    # 1. Максимальная сумма просрочки
    features["max_overdue"] = df.groupby("sk_id_curr")["amt_credit_sum_overdue"].max()
    # 2. Минимальная сумма просрочки
    features["min_overdue"] = df.groupby("sk_id_curr")["amt_credit_sum_overdue"].min()
    # 3. Какую долю суммы от открытого займа просрочил
    total_credit = df.groupby("sk_id_curr")["amt_credit_sum"].sum()
    total_overdue = df.groupby("sk_id_curr")["amt_credit_sum_overdue"].sum()
    features["overdue_proportion"] = total_overdue / total_credit
    # 4. Кол-во кредитов определенного типа

    # 5. Кол-во просрочек кредитов определенного типа

    # 6. Кол-во закрытых кредитов определенного типа
    
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "bureau_features.csv")
    bureau_features.to_csv(output_path, index=False)

    print(bureau_features.head(20))


if __name__ == "__main__":
    main()
