import pandas as pd


def generate_bureau_balance_features(balance_df, bureau_df):
    """
    Генерация признаков из bureau_balance таблицы
    """

    # Копируем индекс
    merged_df = balance_df.merge(
        bureau_df[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="inner"
    )
    features = merged_df[["SK_ID_CURR"]].copy()

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

    bureau_path = "C:\\csv_files\\bureau.csv"
    balance_path = "C:\\csv_files\\bureau_balance.csv"

    bureau_df = pd.read_csv(bureau_path)
    balance_df = pd.read_csv(balance_path)

    bureau_balance_features = generate_bureau_balance_features(balance_df, bureau_df)

    bureau_balance_features.to_csv(
        "src\\app\\modelling\\features\\bureau_balance_features.csv", index=False
    )

    print(bureau_balance_features.head(20))


if __name__ == "__main__":
    main()
