import pickle

from src.app.core.api import Features, ScoringResult, ScoringDecision
from src.app.core.calculator import Calculator


class Model(object):
    """Класс для моделей c расчетом proba и threshold."""

    _threshold = 0.3

    def __init__(self, model_path: str):
        """Создает объект класса."""
        self._calculator = Calculator()
        with open(model_path, 'rb') as pickled_model:
            self._model = pickle.load(pickled_model)

    def get_scoring_result(self, features: Features) -> ScoringResult:
        """Возвращает объект ScoringResult с результатами скоринга."""
        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < self._threshold:
            decision = ScoringDecision.ACCEPTED
            amount = self._calculator.calc_amount(
                proba,
                features,
            )

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Определяет вероятность невозврата займа."""
        df = features.to_dataframe()
        return self._model.predict_proba(df)[0, 1]
