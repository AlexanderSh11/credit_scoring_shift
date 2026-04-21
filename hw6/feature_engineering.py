def feature_engineering(X_train, X_test):
    """Создание новых признаков, которые могут быть полезны"""
    X_train_new_features = X_train.copy()
    X_test_new_features = X_test.copy()
    # Были ли просрочки по кредитам у клиента
    X_train_new_features["had_overdue"] = (
        X_train_new_features["overdue_count_1"] > 0
    ).astype(int)
    X_test_new_features["had_overdue"] = (
        X_test_new_features["overdue_count_1"] > 0
    ).astype(int)
    # Отношение просрочек к общему количеству кредитов
    X_train_new_features["overdue_proportion"] = X_train_new_features[
        "overdue_count_1"
    ] / (X_train_new_features["total_credits_count"] + 1)
    X_test_new_features["overdue_proportion"] = X_test_new_features[
        "overdue_count_1"
    ] / (X_test_new_features["total_credits_count"] + 1)
    # Есть ли открытые кредиты
    X_train_new_features["has_open_credits"] = (
        X_train_new_features["open_credits_count"] > 0
    ).astype(int)
    X_test_new_features["has_open_credits"] = (
        X_test_new_features["open_credits_count"] > 0
    ).astype(int)
    print(
        "Были созданы новые признаки (had_overdue, overdue_proportion, has_open_credits)"
    )
    return X_train_new_features, X_test_new_features
