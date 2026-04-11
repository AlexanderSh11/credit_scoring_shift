import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


class FeatureSelector:
    """
    Класс для отбора признаков
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.importance_df = None

    def select_by_rf(self, X, y, top_n=50, n_estimators=100):
        """Отбор по Random Forest Importance"""

        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )

        self.selected_features = rf_importance.head(top_n).index.tolist()

        self.importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"Отобрано {len(self.selected_features)} признаков по Random Forest")
        return X[self.selected_features]

    def get_selected_features(self):
        return self.selected_features

    def get_importance_df(self):
        return self.importance_df


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

    return df_scaled, scaler


def prepare_data(df, test_size=0.2, top_n=50):
    """
    Общая подготовка данных для всех моделей: фильтрация признаков, заполнение пропусков, кодирование категорий, разделение на train/test
    """
    data_filtered = filter_features(df)
    print(f"После фильтрации: {data_filtered.shape}")

    df_filled = fill_nan_values(data_filtered)
    print(f"После заполнения пропусков: {df_filled.shape}")

    target_col = "target"
    y = df_filled[target_col]
    X = df_filled.drop(columns=[target_col])

    X_encoded, y_encoded = encode_categorical_features(X, y, min_frequency=50)
    print(f"После кодирования: {X_encoded.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    selector = FeatureSelector()
    selector.select_by_rf(X_train, y_train, top_n=top_n)

    return X_train, X_test, y_train, y_test, selector


def train_logistic_regression(X_train, X_test, y_train, y_test, selector):
    """Логистическая регрессия"""
    # Отбор топ-20 по RF важности
    selected_features = selector.get_selected_features()[:20]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_scaled, scaler = scale_features(
        X_train_selected, X_train_selected.columns.tolist()
    )
    X_test_scaled = X_test_selected.copy()
    X_test_scaled[X_train_selected.columns] = scaler.transform(
        X_test_selected[X_train_selected.columns]
    )

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, auc


def train_decision_tree(X_train, X_test, y_train, y_test, selector):
    """Дерево решений"""
    # Отбор топ-30 по RF
    selected_features = selector.get_selected_features()[:30]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, auc


def train_random_forest(X_train, X_test, y_train, y_test, selector):
    """Случайный лес"""
    # Отбор топ-50 по RF важности
    selected_features = selector.get_selected_features()[:50]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, auc


def train_gradient_boosting(X_train, X_test, y_train, y_test, selector):
    """Градиентный бустинг"""
    # Отбор топ-35 по RF важности
    selected_features = selector.get_selected_features()[:35]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = GradientBoostingClassifier(
        n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3
    )
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return model, auc


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

    TEST_SIZE = 0.2

    TOP_N_FEATURES = 50

    # Объединяем по SK_ID_CURR в один датафрейм
    df = application_df.merge(features_df, on="sk_id_curr", how="left")
    # Удаляем test данные
    df = df.dropna(subset=["target"])
    df = df.reset_index(drop=True)
    print(f"Датафрейм с данными для обучения {df.shape}")
    print(df.head())

    X_train, X_test, y_train, y_test, selector = prepare_data(df, test_size=TEST_SIZE, top_n=TOP_N_FEATURES)

    # Обучение всех моделей
    results = {}

    lr_model, lr_auc = train_logistic_regression(X_train, X_test, y_train, y_test, selector)
    results["Logistic Regression"] = lr_auc

    dt_model, dt_auc = train_decision_tree(X_train, X_test, y_train, y_test, selector)
    results["Decision Tree"] = dt_auc

    rf_model, rf_auc = train_random_forest(X_train, X_test, y_train, y_test, selector)
    results["Random Forest"] = rf_auc

    gb_model, gb_auc = train_gradient_boosting(X_train, X_test, y_train, y_test, selector)
    results["Gradient Boosting"] = gb_auc


if __name__ == "__main__":
    main()
