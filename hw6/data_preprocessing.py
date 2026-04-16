from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def drop_columns(df, cols=[], label=""):
    print(f"Удалены {len(cols)} признаков ({label}):")
    print(cols)
    return df.drop(columns=cols)


def fill_nan_values(df):
    """
    Заполнение пропусков, на основе анализа пропусков в датафрейме
    """
    df_filled = df.copy()

    # Заполняем медианой, сохраняет распределение, не создает выбросов
    fill_median = [
        "ext_source_1",
        "ext_source_3",
    ]

    for col in fill_median:
        if col in df_filled.columns and df_filled[col].isnull().any():
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)
            print(f"{col} заполнено медианой ({median_val:.3f})")

    # Заполняем модой в категориальных признаках с небольшим числом пропусков
    fill_mode = [
        "name_type_suite",
    ]

    for col in fill_mode:
        if col in df_filled.columns and df_filled[col].isnull().any():
            mode_val = df_filled[col].mode()[0]
            df_filled[col] = df_filled[col].fillna(mode_val)
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
        if col in df_filled.columns and df_filled[col].isnull().any():
            df_filled[col] = df_filled[col].fillna("unknown")
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
        if col in df_filled.columns:
            before = len(df_filled)
            df_filled = df_filled.dropna(subset=[col])
            after = len(df_filled)
            if before != after:
                print(f"{col} удалено {before - after} строк")

    # Все остальные признаки с пропусками - заполняем 0 (NaN = отсутствие события/характеристики)
    remaining_cols = df_filled.columns[df_filled.isnull().any()].tolist()

    for col in remaining_cols:
        null_count = df_filled[col].isnull().sum()
        df_filled[col] = df_filled[col].fillna(0)
        print(f"{col} заполнено 0 ({null_count} пропусков)")

    return df_filled


def filter_features(df):
    id_cols = ["sk_id_curr", "id_change_delay"]
    cols_to_remove = [col for col in id_cols if col in df.columns]
    df_filtered = drop_columns(df=df, cols=cols_to_remove, label="ID")

    cols_to_remove = []
    for col in df_filtered.columns:
        if df_filtered[col].nunique() == 1:
            cols_to_remove.append(col)
    if cols_to_remove:
        df_filtered = drop_columns(
            df=df_filtered, cols=cols_to_remove, label="Константные признаки"
        )

    isnull_mean = df_filtered.isnull().mean()

    # Удаляем строки с пропусками в признаках где <1% пропусков
    cols_with_few_nulls = isnull_mean[
        (isnull_mean < 0.01) & (isnull_mean > 0)
    ].index.tolist()
    if cols_with_few_nulls:
        before = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=cols_with_few_nulls)
        after = len(df_filtered)
        print(f"Удалено {before - after} строк (в колонках с <1% пропусков)")

    return df_filtered


def encode_categorical_features(X, y, min_frequency=100):
    """
    Кодирование категориальных признаков:
    - Если есть категории, у которых количество строк со значениями в столбце < min_frequency, то их лучше удалить (XNA в code_gender, например)
    - Бинарные (2 значения) - LabelEncoder
    - Остальные - OneHotEncoder (только топ-10 частых, остальные в 'other')
    """
    X_enc = X.copy()
    y_enc = y.copy()

    X_enc = X_enc.reset_index(drop=True)
    y_enc = y_enc.reset_index(drop=True)

    categorical_cols = X.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        # Считаем частоту категорий
        value_counts = X_enc[col].value_counts()
        # Находим редкие категории
        rare_cats = value_counts[value_counts < min_frequency].index.tolist()

        if rare_cats:
            print(f"{col}: редкие категории {rare_cats}")
            # Удаляем строки с редкими категориями
            mask = ~X_enc[col].isin(rare_cats)

            X_enc = X_enc[mask].reset_index(drop=True)
            y_enc = y_enc[mask].reset_index(drop=True)

            print(f"Удалено строк: {(~mask).sum()}")

        n_unique = X_enc[col].nunique()

        le = LabelEncoder()
        ohe = OneHotEncoder(sparse_output=False, drop="first")

        if n_unique == 2:
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))

        elif n_unique > 2:
            # Берем топ-10 частых категорий
            top_cats = X_enc[col].value_counts().head(10).index.tolist()
            # Заменяем редкие на 'other'
            X_temp = X_enc[col].apply(lambda x: x if x in top_cats else "other")

            encoded = ohe.fit_transform(X_temp.to_frame())

            new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]

            for i, new_col in enumerate(new_cols):
                X_enc[new_col] = encoded[:, i]

            X_enc = X_enc.drop(columns=[col])

    return X_enc, y_enc


def remove_outliers(X, cols, multiplier=1.5):
    """
    Удаление выбросов методом IQR
    """
    X_clean = X.copy()

    for col in cols:
        Q1 = X_clean[col].quantile(0.25)
        Q3 = X_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        col_mask = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
        print(f"{col} удалено {col_mask.sum()} выбросов")

    return X_clean


def scale_features(df, cols):
    """
    Масштабирование признаков
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    print("Признаки были масштабированы с помощью StandardScaler")
    return df_scaled, scaler
