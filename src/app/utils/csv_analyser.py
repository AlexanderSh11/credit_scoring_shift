import pandas as pd
import os


class CSVAnalyser:
    """Класс для анализа CSV файлов."""

    def __init__(self, data_path: str):
        """
        Инициализация анализатора CSV файлов.

        :param data_path: путь к директории с CSV файлами
        """
        self.data_path = data_path
        file_path = os.path.join(self.data_path, "HomeCredit_columns_description.csv")
        self.description_df = pd.read_csv(file_path, encoding="cp1251")

    def analyse_file(self, filename: str):
        """
        Анализирует один CSV файл.

        :param filename: имя CSV файла
        """
        file_path = os.path.join(self.data_path, filename)
        df = pd.read_csv(file_path, encoding="cp1251")

        print(f"Название файла: {filename}")
        print(f"Данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

        table_name = filename.replace(".csv", "")
        file_desc = self.description_df[
            self.description_df["Table"].str.contains(table_name)
        ]

        for col in df.columns:
            print(f"Столбец {col}")
            col_desc = file_desc[
                file_desc["Row"].astype(str).str.contains(col, na=False)
            ]
            if not col_desc.empty:
                desc_text = col_desc["Description"].values[0]
                print(f"- Описание: {desc_text}")

                special = col_desc["Special"].values[0]
                if pd.notna(special):
                    print(f"- Особенности: {special}")
            print(f"- Тип данных: {df[col].dtype}")
            print(f"- Уникальных значений: {df[col].nunique()}")
            print(
                f"- Null значений: {df[col].isnull().sum()} ({df[col].isnull().mean():.2%})"
            )

            if df[col].dtype == "object":
                print(f"- Примеры значений: {df[col].dropna().unique()[:5]}")
            else:
                print(f"- Min: {df[col].min()}")
                print(f"- Max: {df[col].max()}")

        df.head()
        return df
