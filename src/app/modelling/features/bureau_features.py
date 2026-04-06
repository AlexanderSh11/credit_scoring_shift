import pandas as pd


def generate_bureau_features(df):
    """
    Генерация признаков из bureau таблицы
    """

    # Копируем индекс
    features = df[["SK_ID_CURR"]].copy()

    # 1. Максимальная сумма просрочки
    features["MAX_OVERDUE"] = df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].max()
    # 2. Минимальная сумма просрочки
    features["MAX_OVERDUE"] = df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].min()
    # 3. Какую долю суммы от открытого займа просрочил
    total_credit = df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    total_overdue = df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].sum()
    features["OVERDUE_PROPORTION"] = total_overdue / total_credit
    # 4. Кол-во кредитов определенного типа

    # 5. Кол-во просрочек кредитов определенного типа

    # 6. Кол-во закрытых кредитов определенного типа
    
    return features


def main():
    """Загрузка данных и генерация признаков для bureau"""

    data_path = "C:\\csv_files\\bureau.csv"

    df = pd.read_csv(data_path)

    bureau_features = generate_bureau_features(df)
    bureau_features.to_csv(
        "src\\app\\modelling\\features\\bureau_features.csv", index=False
    )

    print(bureau_features.head(20))


if __name__ == "__main__":
    main()
