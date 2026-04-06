import pandas as pd


def generate_bureau_features(df):
    """
    Генерация признаков из bureau таблицы
    """

    # Копируем индекс
    features = df[["SK_ID_CURR"]].copy()

    # 1. Максимальная сумма просрочки
    # 2. Минимальная сумма просрочки
    # 3. Какую долю суммы от открытого займа просрочил
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
