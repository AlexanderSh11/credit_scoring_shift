import os
import sys
import numpy as np
from datetime import datetime

# Получаем абсолютный путь к текущему файлу
current_file = os.path.abspath(__file__)
# Переходим в credit_scoring
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
)
sys.path.insert(0, project_root)

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def get_house_columns(df):
    """
    Автоматическое определение колонок с характеристиками дома
    """
    # паттерны для поиска в названиях колонок
    house_patterns = [
        "apartments",
        "basementarea",
        "years_beginexpluatation",
        "years_build",
        "commonarea",
        "elevators",
        "entrances",
        "floorsmax",
        "floorsmin",
        "landarea",
        "livingapartments",
        "livingarea",
        "nonlivingapartments",
        "nonlivingarea",
        "fondkapremont",
        "housetype",
        "totalarea",
        "wallsmaterial",
        "emergencystate",
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
    features["annual_loan_share"] = df["amt_annuity"] / df["amt_income_total"]

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

    return features


def main():
    """Загрузка данных и генерация признаков для application"""

    db_manager = DatabaseManager()
    table_name = "application"
    query = f"""
    SELECT * FROM {table_name}
    """
    df = db_manager.get_df_from_query(query)

    application_features = generate_application_features(df)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "application_features.csv")
    application_features.to_csv(output_path, index=False)

    print(application_features.head(20))


if __name__ == "__main__":
    main()
