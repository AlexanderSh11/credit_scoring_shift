import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_loading import (
    load_application_data,
    load_features_from_csv,
)
from data_preprocessing import (
    encode_categorical_features,
    fill_nan_values,
    filter_features,
)
from feature_engineering import feature_engineering
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


FEATURES_DIR = PROJECT_DIR / "src" / "app" / "modelling" / "features" / "data"
FEATURES_FILES = {
    "application": "application_features.csv",
    "bureau": "bureau_features.csv",
    "bureau_balance": "bureau_balance_features.csv",
}

TEST_SIZE = 0.2

TOP_N_FEATURES = 50

MIN_FREQUENCY_IN_CAT_FEATURE = 5

MODELS_DIR = PROJECT_DIR / "hw6" / "models"

DATA_PATH_DIR = PROJECT_DIR / "hw6" / "data"


def sync_indexes(X_train, X_test, y_train, y_test):
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    return y_train, y_test


def prepare_data(df, test_size=0.2, top_n=50, min_frequency=100):
    """
    Общая подготовка данных для всех моделей: разделение на train/test, фильтрация признаков, заполнение пропусков, кодирование категорий
    """
    target_col = "target"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    X_train_filtered, X_test_filtered = filter_features(X_train, X_test)
    print(
        f"После фильтрации: Train: {X_train_filtered.shape}, Test: {X_test_filtered.shape}"
    )
    # Синхронизируем индексы X с y
    y_train, y_test = sync_indexes(X_train_filtered, X_test_filtered, y_train, y_test)

    X_train_filled, X_test_filled = fill_nan_values(X_train_filtered, X_test_filtered)
    print(
        f"После заполнения пропусков: {X_train_filled.shape}, Test: {X_test_filled.shape}"
    )
    y_train, y_test = sync_indexes(X_train_filled, X_test_filled, y_train, y_test)

    X_train_new_features, X_test_new_features = feature_engineering(
        X_train_filled, X_test_filled
    )
    y_train, y_test = sync_indexes(
        X_train_new_features, X_test_new_features, y_train, y_test
    )

    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = (
        encode_categorical_features(
            X_train_new_features,
            X_test_new_features,
            y_train,
            y_test,
            min_frequency=min_frequency,
        )
    )
    print(f"После кодирования: {X_train_encoded.shape}, Test: {X_test_encoded.shape}")

    selector = FeatureSelector()
    selector.select_by_rf(X_train_encoded, y_train_encoded, top_n=top_n)
    print("Выбранные признаки с помощью Random Forest")
    print(selector.get_selected_features())

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, selector


def main():
    db_manager = DatabaseManager()
    application_df = load_application_data(db_manager)

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

    X_train, X_test, y_train, y_test, selector = prepare_data(
        df,
        test_size=TEST_SIZE,
        top_n=TOP_N_FEATURES,
        min_frequency=MIN_FREQUENCY_IN_CAT_FEATURE,
    )

    # Обучение всех моделей
    trainer = ModelTrainer(
        random_state=42, models_dir=MODELS_DIR, data_path_dir=DATA_PATH_DIR
    )
    trainer.train_all(X_train, X_test, y_train, y_test, selector, top_n=20)


if __name__ == "__main__":
    main()
