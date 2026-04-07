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


def generate_bureau_balance_features(balance_df, bureau_df):
    """
    Генерация признаков из bureau_balance таблицы
    """

    # Соединяем bureau_balance с bureau для получения sk_id_curr
    merged_df = balance_df.merge(
        bureau_df[["sk_id_bureau", "sk_id_curr"]], on="sk_id_bureau", how="inner"
    )
    features = merged_df[["sk_id_curr", "sk_id_bureau"]].copy()

    # 1. Кол-во открытых кредитов
    # 2. Кол-во закрытых кредитов
    # 3. Кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    # 4. Кол-во кредитов
    # 5. Доля закрытых кредитов
    # 6. Доля открытых кредитов
    # 7. Доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    # 8. Интервал между последним закрытым кредитом и текущей заявкой
    # 9. Интервал между взятием последнего активного займа и текущей заявкой
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

    bureau_balance_features = generate_bureau_balance_features(balance_df=balance_df, bureau_df=bureau_df)
    
    current_dir = os.path.dirname(current_file)
    output_path = os.path.join(current_dir, "bureau_balance_features.csv")
    bureau_balance_features.to_csv(output_path, index=False)

    print(bureau_balance_features.head(20))


if __name__ == "__main__":
    main()
