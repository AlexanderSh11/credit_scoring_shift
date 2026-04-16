def feature_engineering(df):
    """Создание новых признаков, которые могут быть полезны"""
    df_new_features = df.copy()
    # Были ли просрочки по кредитам у клиента
    df_new_features["had_overdue"] = (df_new_features["overdue_count_1"] > 0).astype(
        int
    )
    # Отношение просрочек к общему количеству кредитов
    df_new_features["overdue_proportion"] = df_new_features["overdue_count_1"] / (
        df_new_features["total_credits_count"] + 1
    )
    # Есть ли открытые кредиты
    df_new_features["has_open_credits"] = (
        df_new_features["open_credits_count"] > 0
    ).astype(int)
    print(
        "Были созданы новые признаки (had_overdue, overdue_proportion, has_open_credits)"
    )
    return df_new_features
