from src.app.core.api import Features


class Calculator:
    def calc_amount(
        self,
        proba: str,
        features: Features,
    ) -> int:
        """Функция принимает на вход вероятность дефолта и признаки и расчитывает одобренную сумму."""
        if proba < 0.10 and features.income_category == '> 50_000':
            return 500_000
        if features.age_category == 'other':
            return 200_000
        return 100_000