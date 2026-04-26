from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def drop_columns(df, cols=[], label=""):
    print(f"Удалены {len(cols)} признаков ({label}):")
    print(cols)
    return df.drop(columns=cols)


def fill_nan_values(X_train, X_test):
    """
    Заполнение пропусков, на основе анализа пропусков в датафрейме
    """
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()

    # Заполняем медианой на train, сохраняет распределение, не создает выбросов
    fill_median = [
        "ext_source_1",
        "ext_source_3",
    ]

    for col in fill_median:
        if col in X_train_filled.columns:
            median_val = X_train_filled[col].median()
            X_train_filled[col] = X_train_filled[col].fillna(median_val)
            X_test_filled[col] = X_test_filled[col].fillna(median_val)
            print(f"{col} заполнено медианой ({median_val:.3f})")

    # Заполняем модой на train в категориальных признаках с небольшим числом пропусков
    fill_mode = [
        "name_type_suite",
    ]

    for col in fill_mode:
        if col in X_train_filled.columns:
            mode_val = X_train_filled[col].mode()[0]
            X_train_filled[col] = X_train_filled[col].fillna(mode_val)
            X_test_filled[col] = X_test_filled[col].fillna(mode_val)
            print(f"{col} заполнено модой ('{mode_val}')")

    # Заполняем unknown в категориальных признаках, где много пропусков
    fill_unknown = [
        "occupation_type",
        "fondkapremont_mode",
        "housetype_mode",
        "wallsmaterial_mode",
        "emergencystate_mode",
    ]

    for col in fill_unknown:
        if col in X_train_filled.columns:
            X_train_filled[col] = X_train_filled[col].fillna("unknown")
            X_test_filled[col] = X_test_filled[col].fillna("unknown")
            print(f"{col} заполнено 'unknown'")

    # Удаляем строки с пропусками в этих колонках
    drop_rows_cols = [
        "amt_req_credit_bureau_hour",
        "amt_req_credit_bureau_day",
        "amt_req_credit_bureau_week",
        "amt_req_credit_bureau_mon",
        "amt_req_credit_bureau_qrt",
        "amt_req_credit_bureau_year",
    ]

    for col in drop_rows_cols:
        if col in X_train_filled.columns:
            before_train = len(X_train_filled)
            X_train_filled = X_train_filled.dropna(subset=[col])
            after_train = len(X_train_filled)
            if before_train != after_train:
                print(f"Train: {col} удалено {before_train - after_train} строк")

            before_test = len(X_test_filled)
            X_test_filled = X_test_filled.dropna(subset=[col])
            after_test = len(X_test_filled)
            if before_test != after_test:
                print(f"Test: {col} удалено {before_test - after_test} строк")

    # Все остальные признаки с пропусками - заполняем 0 (NaN = отсутствие события/характеристики)
    remaining_cols_train = X_train_filled.columns[
        X_train_filled.isnull().any()
    ].tolist()
    remaining_cols_test = X_test_filled.columns[X_test_filled.isnull().any()].tolist()

    for col in remaining_cols_train:
        null_count = X_train_filled[col].isnull().sum()
        X_train_filled[col] = X_train_filled[col].fillna(0)
        print(f"Train: {col} заполнено 0 ({null_count} пропусков)")

    for col in remaining_cols_test:
        null_count = X_test_filled[col].isnull().sum()
        X_test_filled[col] = X_test_filled[col].fillna(0)
        print(f"Test: {col} заполнено 0 ({null_count} пропусков)")

    return X_train_filled, X_test_filled


def filter_features(X_train, X_test):
    id_cols = ["sk_id_curr", "id_change_delay"]
    cols_to_remove = [col for col in id_cols if col in X_train.columns]
    X_train_filtered = drop_columns(df=X_train, cols=cols_to_remove, label="ID")
    X_test_filtered = drop_columns(df=X_test, cols=cols_to_remove, label="ID")

    cols_to_remove = []
    for col in X_train_filtered.columns:
        if X_train_filtered[col].nunique() == 1:
            cols_to_remove.append(col)
    if cols_to_remove:
        X_train_filtered = drop_columns(
            df=X_train_filtered, cols=cols_to_remove, label="Константные признаки"
        )
        X_test_filtered = drop_columns(
            df=X_test_filtered, cols=cols_to_remove, label="Константные признаки"
        )

    # Определяем колонки с малым количеством пропусков в train
    isnull_mean = X_train_filtered.isnull().mean()
    cols_with_few_nulls = isnull_mean[
        (isnull_mean < 0.01) & (isnull_mean > 0)
    ].index.tolist()

    if cols_with_few_nulls:
        # Удаляем строки с пропусками в этих колонках из train
        before_train = len(X_train_filtered)
        X_train_filtered = X_train_filtered.dropna(subset=cols_with_few_nulls)
        after_train = len(X_train_filtered)
        print(
            f"Из train удалено {before_train - after_train} строк (в колонках с <1% пропусков)"
        )

        # Для test удаляем только те строки, где есть пропуски в этих же колонках
        before_test = len(X_test_filtered)
        X_test_filtered = X_test_filtered.dropna(subset=cols_with_few_nulls)
        after_test = len(X_test_filtered)
        print(f"Из test удалено строк: {before_test - after_test}")

    return X_train_filtered, X_test_filtered


def encode_categorical_features(X_train, X_test, y_train, y_test, min_frequency=100):
    """
    Кодирование категориальных признаков:
    - Если есть категории, у которых количество строк со значениями в столбце < min_frequency, то их лучше удалить (XNA в code_gender, например)
    - Бинарные (2 значения) - LabelEncoder
    - Остальные - OneHotEncoder (только топ-10 частых, остальные в 'other')
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    y_train_enc = y_train.copy()
    y_test_enc = y_test.copy()

    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        # Считаем частоту категорий
        value_counts = X_train_enc[col].value_counts()
        # Находим редкие категории
        rare_cats = value_counts[value_counts < min_frequency].index.tolist()

        if rare_cats:
            print(f"{col}: редкие категории {rare_cats}")
            # Заменяем редкие категории на 'other'
            X_train_enc[col] = X_train_enc[col].apply(lambda x: 'other' if x in rare_cats else x)
            X_test_enc[col] = X_test_enc[col].apply(lambda x: 'other' if x in rare_cats else x)

        n_unique = X_train_enc[col].nunique()

        le = LabelEncoder()
        ohe = OneHotEncoder(sparse_output=False, drop="first")

        if n_unique == 2:
            le.fit(X_train_enc[col].astype(str))
            X_train_enc[col] = le.transform(X_train_enc[col].astype(str))
            X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

        elif n_unique > 2:
            # Берем топ-10 частых категорий на train
            top_cats = X_train_enc[col].value_counts().head(10).index.tolist()
            # Заменяем редкие на 'other'
            X_train_temp = X_train_enc[col].apply(lambda x: x if x in top_cats else "other").to_frame()
            ohe.fit(X_train_temp)

            # Применяем к train
            X_train_encoded = ohe.transform(X_train_temp)
            new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
            for i, new_col in enumerate(new_cols):
                X_train_enc[new_col] = X_train_encoded[:, i]
            
            # Применяем к test
            X_test_temp = X_test_enc[col].apply(lambda x: x if x in top_cats else "other").to_frame()
            X_test_encoded = ohe.transform(X_test_temp)
            for i, new_col in enumerate(new_cols):
                X_test_enc[new_col] = X_test_encoded[:, i]
            
            # Удаляем исходную колонку
            X_train_enc = X_train_enc.drop(columns=[col])
            X_test_enc = X_test_enc.drop(columns=[col])

    return X_train_enc, X_test_enc, y_train_enc, y_test_enc


def scale_features(df, cols):
    """
    Масштабирование признаков
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    print("Признаки были масштабированы с помощью StandardScaler")
    return df_scaled, scaler
