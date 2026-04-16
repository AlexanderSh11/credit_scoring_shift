import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.app.utils.db_manager import DatabaseManager  # noqa: E402
from .data_loading import (  # noqa: E402
    load_application_data,
    load_features_from_csv,
)
from .data_preprocessing import (  # noqa: E402
    encode_categorical_features,
    fill_nan_values,
    filter_features,
)
from .feature_engineering import feature_engineering  # noqa: E402
from .feature_selector import FeatureSelector  # noqa: E402
from .model_trainer import ModelTrainer  # noqa: E402


def prepare_data(df, test_size=0.2, top_n=50, min_frequency=100):
    """
    Общая подготовка данных для всех моделей: фильтрация признаков, заполнение пропусков, кодирование категорий, разделение на train/test
    """
    data_filtered = filter_features(df)
    print(f"После фильтрации: {data_filtered.shape}")

    df_filled = fill_nan_values(data_filtered)
    print(f"После заполнения пропусков: {df_filled.shape}")

    df_new_features = feature_engineering(df_filled)

    target_col = "target"
    y = df_new_features[target_col]
    X = df_new_features.drop(columns=[target_col])

    X_encoded, y_encoded = encode_categorical_features(
        X, y, min_frequency=min_frequency
    )
    print(f"После кодирования: {X_encoded.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    selector = FeatureSelector()
    selector.select_by_rf(X_train, y_train, top_n=top_n)
    print("Выбранные признаки с помощью Random Forest")
    print(selector.get_selected_features())

    return X_train, X_test, y_train, y_test, selector


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

    MIN_FREQUENCY_IN_CAT_FEATURE = 5

    MODELS_DIR = PROJECT_DIR / "hw6" / "models"

    DATA_PATH_DIR = PROJECT_DIR / "hw6" / "data"

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
