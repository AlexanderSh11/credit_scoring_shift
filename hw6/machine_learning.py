import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

from src.app.utils.db_manager import DatabaseManager  # noqa: E402


def load_application_data(db_manager):
    application_table_name = "application"
    query = f"""
    SELECT *
    FROM {application_table_name}
    """
    return db_manager.get_df_from_query(query)


def main():
    db_manager = DatabaseManager()
    application_df = load_application_data(db_manager)
    print(application_df.head())


if __name__ == "__main__":
    main()
