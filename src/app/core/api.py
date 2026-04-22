from dataclasses import dataclass
from enum import Enum, auto


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
    days_registration: int = 0
    days_birth: int = 0
    days_id_publish: int = 0
    annuity_to_income_proportion: float = 0.0
    interest_rate: float = 0.0
    days_employed: int = 0
    amt_annuity: float = 0.0
