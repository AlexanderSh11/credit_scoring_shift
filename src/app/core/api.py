from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd


class ScoringDecision(Enum):
    """Возможные решения модели."""

    ACCEPTED = auto()
    DECLINED = auto()


@dataclass
class ScoringResult:
    """Класс, содержащий результаты скоринга."""

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


@dataclass
class Features:
    """Фичи для принятия решения об одобрении."""

    weighted_ext_score: float = 0.0
    ext_source_3: float = 0.0
    ext_source_2: float = 0.0
    days_registration: float = 0.0
    days_birth: float = 0.0
    days_id_publish: float = 0.0
    annuity_to_income_proportion: float = 0.0
    interest_rate: float = 0.0
    days_employed: float = 0.0
    amt_annuity: float = 0.0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Создает Features из DataFrame"""
        return cls(
            weighted_ext_score=df["weighted_ext_score"].iloc[0],
            ext_source_3=df["ext_source_3"].iloc[0],
            ext_source_2=df["ext_source_2"].iloc[0],
            days_registration=df["days_registration"].iloc[0],
            days_birth=df["days_birth"].iloc[0],
            days_id_publish=df["days_id_publish"].iloc[0],
            annuity_to_income_proportion=df["annuity_to_income_proportion"].iloc[0],
            interest_rate=df["interest_rate"].iloc[0],
            days_employed=df["days_employed"].iloc[0],
            amt_annuity=df["amt_annuity"].iloc[0],
        )
