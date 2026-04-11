import sys
from pathlib import Path
import pandas as pd

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


def main():
    db_manager = DatabaseManager()
    application_df = load_application_data(db_manager)
    print(application_df.head())

    FEATURES_DIR = PROJECT_DIR / "src" / "app" / "modelling" / "features"
    FEATURES_FILES = {
        "application": "application_features.csv",
        "bureau": "bureau_features.csv",
        "bureau_balance": "bureau_balance_features.csv",
    }
    features_df = load_features_from_csv(
        features_dir=FEATURES_DIR, features_files=FEATURES_FILES
    )
    print(features_df.head())


if __name__ == "__main__":
    main()
