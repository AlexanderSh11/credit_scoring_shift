import os
import sys
import numpy as np
from typing import List
from datetime import datetime

import pandas as pd

# Получаем абсолютный путь к текущему файлу
current_file = os.path.abspath(__file__)
# Переходим в credit_scoring
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
)
sys.path.insert(0, project_root)

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def get_house_columns(df: pd.DataFrame) -> List[str]:
    """
    Автоматическое определение колонок с характеристиками дома
    """
    # паттерны для поиска в названиях колонок
    house_suffixes = ("_avg", "_mode", "_medi")

    house_cols = []

    for col in df.columns:
        if any(col.endswith(suffix) for suffix in house_suffixes):
            house_cols.append(col)

    return house_cols


def find_interest_rate(
    PV: float, P: float, n: float, low: float = 0.0001, high: float = 0.1
) -> float:
    """
    :param PV: сумма кредита
    :param P: ежемесячный платеж
    :param n: срок в месяцах
    :param low: нижняя граница ставки
    :param high: верхняя граница ставки
    """
    if pd.isna(PV) or pd.isna(P) or pd.isna(n):
        return np.nan
    if PV <= 0 or P <= 0 or n <= 0:
        return np.nan

    # Платежи за срок n должны покрыть кредит
    if P * n <= PV:
        return np.nan

    # Метод бисекции
    low, high = 0.0001, 0.1

    for _ in range(50):
        # на каждом шаге делит интервал пополам и выбирает ту половину, где рассчитанный платеж отличается от заданного в нужную сторону
        r = (low + high) / 2
        try:
            # IRR - это ставка r, при которой чистая приведенная стоимость (NPV) = 0
            # NPV = PV - P/(1+r) - P/(1+r)^2 - ... - P/(1+r)^n = 0
            # или PV = P * (1 - (1+r)^(-n)) / r
            # P = PV * r / (1 - (1+r)^(-n))
            calculated_P = PV * r / (1 - (1 + r) ** (-n))
        except Exception:
            return np.nan

        if calculated_P > P:
            high = r
        else:
            low = r

    monthly_rate = (low + high) / 2
    annual_rate = (1 + monthly_rate) ** 12 - 1
    result = annual_rate * 100

    return result


def generate_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков из application таблицы
    """

    # Копируем индекс
    features = df[["sk_id_curr"]].copy()

    # 1. Кол-во документов (FLAG_DOCUMENT_2 ... FLAG_DOCUMENT_21)
    doc_flags = [col for col in df.columns if col.startswith("flag_document_")]
    features["document_count"] = df[doc_flags].sum(axis=1)

    # 2. Есть ли полная информация о доме. Найдите все колонки, которые описывают характеристики дома и посчитайте кол-во непустых характеристик.
    # Колонки, описывающие характеристики дома
    house_cols = get_house_columns(df)
    # Считаем количество непустых характеристик
    non_null_count = df[house_cols].notna().sum(axis=1)
    # Если кол-во пропусков меньше 30, то значение признака 1. Иначе 0
    filled = non_null_count / len(house_cols)
    features["full_house_info"] = (filled > 0.7).astype(int)

    # 3. Кол-во полных лет
    features["age_full_years"] = (-df["days_birth"] // 365.2425).astype(int)

    # 4. Сколько лет назад был сменён документ
    # Согласно описанию: DAYS_ID_PUBLISH - сколько дней до заявки сменили документ
    features["years_since_id_change"] = (-df["days_id_publish"] // 365.2425).astype(int)
    # Год смены документа
    current_year = datetime.now().year
    features["id_change_year"] = current_year - features["years_since_id_change"]

    # 5. В каком возрасте клиент сменил документ
    features["age_at_id_change"] = (
        (-df["days_birth"] - df["days_id_publish"]) // 365.2425
    ).astype(int)

    # 6. Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет
    expected_ages = [14, 20, 45]
    delay_conditions = []

    for exp_age in expected_ages:
        condition = (features["age_full_years"] >= exp_age) & (
            features["age_at_id_change"] != exp_age
        )
        delay_conditions.append(condition)

    features["id_change_delay"] = np.any(delay_conditions, axis=0).astype(int)

    # 7. Доля денег которые клиент отдает на займ за год
    features["annuity_to_income_proportion"] = (
        df["amt_annuity"] / df["amt_income_total"]
    )

    # 8. Среднее кол-во детей в семье на одного взрослого
    # CNT_CHILDREN - количество детей, CNT_FAM_MEMBERS - размер семьи
    n_adults = df["cnt_fam_members"] - df["cnt_children"]
    features["children_per_adult"] = df["cnt_children"] / n_adults

    # 9. Средний доход на ребенка
    features["income_per_child"] = df["amt_income_total"] / df["cnt_children"]
    features["income_per_child"] = features["income_per_child"].replace(
        [np.inf, -np.inf], np.nan
    )

    # 10. Средний доход на взрослого
    features["income_per_adult"] = df["amt_income_total"] / n_adults

    # 11. Взвешенный скор внешних источников. Подумайте какие веса им задать и поясните свой выбор.
    # Произвольный выбор весов (например, равных) не имеет обоснования. В тренировочных данных есть целевая переменная TARGET
    # Логично предположить, что источник, имеющий более сильную связь с просрочкой, должен иметь больший вес
    # Веса рассчитаны на основе корреляции с целевой переменной TARGET в тренировочных данных
    # Чем выше модуль корреляции, тем больше вес
    ext_sources = ["ext_source_1", "ext_source_2", "ext_source_3"]
    # Получаем train по наличию target
    train_df = df[df["target"].notna()]
    # Считаем веса только на train
    correlations = train_df[
        ["ext_source_1", "ext_source_2", "ext_source_3", "target"]
    ].corr()
    weights = abs(correlations["target"].drop("target"))
    weights = weights / weights.sum()
    features["weighted_ext_score"] = (df[ext_sources] * weights).sum(axis=1)

    # 12. Поделим людей на группы в зависимости от пола и образования. В каждой группе посчитаем средний доход. Сделаем признак разница между средним доходом в группе и доходом заявителя
    # Группируем по полу (CODE_GENDER) и образованию (NAME_EDUCATION_TYPE)
    gender_educ_group = (
        df["code_gender"].astype(str) + "_" + df["name_education_type"].astype(str)
    )

    # Считаем средний доход в группе
    group_avg_income = df.groupby(gender_educ_group)["amt_income_total"].transform(
        "mean"
    )

    # Разница между доходом клиента и средним по группе
    features["income_diff_from_group_avg"] = df["amt_income_total"] - group_avg_income

    # 13. Посчитать процентную ставку (*) (можно использовать все таблицы, например previous_application)
    # Из таблицы previous_application найдем все уникальные сроки прошлых кредитов клиентов
    possible_months = sorted(df["cnt_payment"].dropna().unique())

    def calc_rate(row):
        PV = row["amt_credit"]
        P = row["amt_annuity"]

        for n in possible_months:
            # Поиск ставки
            rate = find_interest_rate(PV, P, n, low=0.0001, high=0.1)
            if not pd.isna(rate) and 0 < rate <= 100:
                return rate
        return np.nan

    features["interest_rate"] = df.apply(calc_rate, axis=1)

    return features


def main():
    """Загрузка данных и генерация признаков для application"""

    db_manager = DatabaseManager()
    application_table_name = "application"
    previous_table_name = "previous_application"
    query = f"""
    SELECT 
        a.*,
        p.CNT_PAYMENT
    FROM {application_table_name} a
    LEFT JOIN (
        SELECT DISTINCT ON (SK_ID_CURR) SK_ID_CURR, CNT_PAYMENT
        FROM {previous_table_name}
        WHERE CNT_PAYMENT IS NOT NULL AND CNT_PAYMENT > 0
        ORDER BY SK_ID_CURR DESC
    ) p ON a.SK_ID_CURR = p.SK_ID_CURR
    """
    df = db_manager.get_df_from_query(query)

    application_features = generate_application_features(df)
    current_dir = os.path.dirname(current_file)
    output_path = os.path.join(current_dir, "application_features.csv")
    application_features.to_csv(output_path, index=False)

    print(application_features.head(20))


if __name__ == "__main__":
    main()
