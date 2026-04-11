import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def load_application_data(db_manager):
    """
    Загрузка train/test данных из БД
    """
    application_table_name = "application"
    query = f"""
    SELECT *
    FROM {application_table_name}
    """
    return db_manager.get_df_from_query(query)


def load_features_from_csv(features_dir=None, features_files=[]):
    """
    Загрузка сгенерированных признаков из CSV файлов
    """
    features_dir = Path(features_dir)

    if not features_dir.exists():
        raise FileNotFoundError(
            f"Директория с файлами признаков не найдена: {features_dir}"
        )

    dataframes = {}

    for key, filename in features_files.items():
        file_path = features_dir / filename
        if file_path.exists():
            dataframes[key] = pd.read_csv(file_path)
        else:
            print(f"Файл не найден: {filename}")

    first_key = list(dataframes.keys())[0]
    features_df = dataframes[first_key].copy()

    for key, df in list(dataframes.items())[1:]:
        # Объединяем по SK_ID_CURR
        features_df = features_df.merge(df, on="sk_id_curr", how="left")

    return features_df


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
    id_cols = ["sk_id_curr"]
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

    # Удаляем признаки с >70% пропусков
    cols_to_remove = isnull_mean[isnull_mean > 0.7].index.tolist()
    if cols_to_remove:
        df_filtered = drop_columns(
            df=df_filtered, cols=cols_to_remove, label="Признаки с >70% пропусков"
        )

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


def select_features_by_importance(X, y, top_n=50):
    """
    Отбор топ-N признаков по важности с помощью Random Forest
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    # Выбираем топ-N по RF важности
    selected_features = rf_importance.head(top_n).index.tolist()
    X_selected = X[selected_features]

    print(f"Отобрано {len(selected_features)} признаков")

    for i, feature in enumerate(selected_features[:10], 1):
        rf_val = rf_importance[feature]
        print(f"  {i:2d}. {feature:<35} RF={rf_val:.4f}")

    # Сохраняем важность всех признаков
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "rf_importance": rf_importance.values,
        }
    ).sort_values("rf_importance", ascending=False)

    return X_selected, selected_features, importance_df


def features_selection(df, top_n=50, test_size=0.2):
    data = df.copy()

    data_filtered = filter_features(df=data)
    print(f"Датафрейм с очищенными данными {data_filtered.shape}")

    df_filled = fill_nan_values(df=data_filtered)
    print(f"Датафрейм с заполненными данными {df_filled.shape}")

    target_col = "target"
    y = df_filled[target_col]
    X = df_filled.drop(columns=[target_col])

    X_encoded, y_encoded = encode_categorical_features(X, y, min_frequency=50)
    print(f"Датафрейм с закодированными признаками {X_encoded.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    X_train_selected, selected_features, importance_df = select_features_by_importance(
        X_train, y_train, top_n
    )
    X_test_selected = X_test[selected_features]
    print(
        f"Датафрейм с выбранными признаками: Train={X_train_selected.shape}, Test={X_test_selected.shape}"
    )
    print(importance_df)

    return X_train_selected, X_test_selected, y_train, y_test, selected_features


def main():
    db_manager = DatabaseManager()
    application_df = load_application_data(db_manager)

    FEATURES_DIR = PROJECT_DIR / "src" / "app" / "modelling" / "features"
    FEATURES_FILES = {
        "application": "application_features.csv",
        "bureau": "bureau_features.csv",
        "bureau_balance": "bureau_balance_features.csv",
    }
    features_df = load_features_from_csv(
        features_dir=FEATURES_DIR, features_files=FEATURES_FILES
    )

    # Объединяем по SK_ID_CURR в один датафрейм
    df = application_df.merge(features_df, on="sk_id_curr", how="left")
    # Удаляем test данные
    df = df.dropna(subset=["target"])
    df = df.reset_index(drop=True)
    print(f"Датафрейм с данными для обучения {df.shape}")
    print(df.head())

    X_train, X_test, y_train, y_test, selected_features = features_selection(
        df, top_n=50
    )


if __name__ == "__main__":
    main()
