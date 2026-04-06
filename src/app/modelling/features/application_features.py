import pandas as pd
import numpy as np
from datetime import datetime


def get_house_columns(df):
    """
    Автоматическое определение колонок с характеристиками дома
    """
    # паттерны для поиска в названиях колонок
    house_patterns = [
        "APARTMENTS",
        "BASEMENTAREA",
        "YEARS_BEGINEXPLUATATION",
        "YEARS_BUILD",
        "COMMONAREA",
        "ELEVATORS",
        "ENTRANCES",
        "FLOORSMAX",
        "FLOORSMIN",
        "LANDAREA",
        "LIVINGAPARTMENTS",
        "LIVINGAREA",
        "NONLIVINGAPARTMENTS",
        "NONLIVINGAREA",
        "FONDKAPREMONT",
        "HOUSETYPE",
        "TOTALAREA",
        "WALLSMATERIAL",
        "EMERGENCYSTATE",
    ]

    house_cols = []

    for col in df.columns:
        if any(pattern in col for pattern in house_patterns):
            house_cols.append(col)

    return house_cols


def generate_application_features(df):
    """
    Генерация признаков из application таблицы
    """

    # Копируем индекс
    features = df[["SK_ID_CURR"]].copy()

    # 1. Кол-во документов (FLAG_DOCUMENT_2 ... FLAG_DOCUMENT_21)
    doc_flags = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
    features["DOCUMENT_COUNT"] = df[doc_flags].sum(axis=1)

    # 2. Есть ли полная информация о доме. Найдите все колонки, которые описывают характеристики дома и посчитайте кол-во непустых характеристик.
    # Колонки, описывающие характеристики дома
    house_cols = get_house_columns(df)
    # Считаем количество непустых характеристик
    non_null_count = df[house_cols].notna().sum(axis=1)
    # Если кол-во пропусков меньше 30, то значение признака 1. Иначе 0
    filled = non_null_count / len(house_cols)
    features["FULL_HOUSE_INFO"] = (filled > 0.7).astype(int)

    # 3. Кол-во полных лет
    features["AGE_FULL_YEARS"] = (-df["DAYS_BIRTH"] // 365.2425).astype(int)

    # 4. Сколько лет назад был сменён документ
    # Согласно описанию: DAYS_ID_PUBLISH - сколько дней до заявки сменили документ
    features["YEARS_SINCE_ID_CHANGE"] = (-df["DAYS_ID_PUBLISH"] // 365.2425).astype(int)
    # Год смены документа
    current_year = datetime.now().year
    features["ID_CHANGE_YEAR"] = current_year - features["YEARS_SINCE_ID_CHANGE"]

    # 5. В каком возрасте клиент сменил документ
    features["AGE_AT_ID_CHANGE"] = (
        (-df["DAYS_BIRTH"] - df["DAYS_ID_PUBLISH"]) // 365.2425
    ).astype(int)

    # 6. Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет
    expected_ages = [14, 20, 45]
    delay_conditions = []

    for exp_age in expected_ages:
        condition = (features["AGE_FULL_YEARS"] >= exp_age) & (
            features["AGE_AT_ID_CHANGE"] != exp_age
        )
        delay_conditions.append(condition)

    features["ID_CHANGE_DELAY"] = np.any(delay_conditions, axis=0).astype(int)

    # 7. Доля денег которые клиент отдает на займ за год
    features["ANNUAL_LOAN_SHARE"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # 8. Среднее кол-во детей в семье на одного взрослого
    # CNT_CHILDREN - количество детей, CNT_FAM_MEMBERS - размер семьи
    n_adults = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    features["CHILDREN_PER_ADULT"] = df["CNT_CHILDREN"] / n_adults

    # 9. Средний доход на ребенка
    features["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / df["CNT_CHILDREN"]
    features["INCOME_PER_CHILD"] = features["INCOME_PER_CHILD"].replace(
        [np.inf, -np.inf], np.nan
    )

    # 10. Средний доход на взрослого
    features["INCOME_PER_ADULT"] = df["AMT_INCOME_TOTAL"] / n_adults

    # 11. Взвешенный скор внешних источников. Подумайте какие веса им задать и поясните свой выбор.

    # 12. Поделим людей на группы в зависимости от пола и образования. В каждой группе посчитаем средний доход. Сделаем признак разница между средним доходом в группе и доходом заявителя
    # Группируем по полу (CODE_GENDER) и образованию (NAME_EDUCATION_TYPE)
    gender_educ_group = (
        df["CODE_GENDER"].astype(str) + "_" + df["NAME_EDUCATION_TYPE"].astype(str)
    )

    # Считаем средний доход в группе
    group_avg_income = df.groupby(gender_educ_group)["AMT_INCOME_TOTAL"].transform(
        "mean"
    )

    # Разница между доходом клиента и средним по группе
    features["INCOME_DIFF_FROM_GROUP_AVG"] = df["AMT_INCOME_TOTAL"] - group_avg_income

    # 13. Посчитать процентную ставку (*) (можно использовать все таблицы, например previous_application)

    return features


def main():
    """Загрузка данных и генерация признаков для application"""

    train_path = "C:\\csv_files\\application_train.csv"

    train_df = pd.read_csv(train_path)

    train_features = generate_application_features(train_df)
    train_features.to_csv(
        "src\\app\\modelling\\features\\application_features.csv", index=False
    )

    print(train_features.head(20))


if __name__ == "__main__":
    main()
