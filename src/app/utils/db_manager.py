import psycopg2
import pandas as pd
from datetime import datetime

from ...config.config import DB_ARGS


def calc_duration(func):
    def wrapper(*args, **kwargs):
        """Обертка над функциями для подсчета времени их работы.

        :return: Результат обернутой функции.
        """
        # Вызываем переданную функцию и считаем затраты времени
        time_before = datetime.now()
        result = func(*args, **kwargs)
        time_after = datetime.now()
        time_delta = time_after - time_before
        print(f"Время выполнения: {str(time_delta).split('.')[0]}")

        return result

    return wrapper


class DatabaseManager:
    """Класс для управления операциями с базой данных."""

    def __init__(self):
        """Инициализация менеджера БД."""
        self.db_args = DB_ARGS
        self._connection = None
        self._cursor = None

    def _connect(self):
        """Устанавливает соединение с БД."""
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(**self.db_args)
                self._cursor = self._connection.cursor()
        except psycopg2.Error as e:
            raise ConnectionError(f"Connection error: {e}")

    def _disconnect(self):
        """Закрывает соединение с БД."""
        try:
            if self._cursor and not self._cursor.closed:
                self._cursor.close()
            if self._connection and not self._connection.closed:
                self._connection.close()
        except psycopg2.Error as e:
            print(f"Error closing connection: {e}")
        finally:
            self._cursor = None
            self._connection = None

    @calc_duration
    def send_sql_query(self, query: str):
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.
        """
        self._connect()
        try:
            self._cursor.execute(query)
            self._connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            self._disconnect()

    @calc_duration
    def get_df_from_query(self, query: str) -> pd.DataFrame:
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.

        :return df: датафрейм с результатом.
        """
        self._connect()
        df = pd.read_sql(query, self._connection)
        self._disconnect()
        return df
