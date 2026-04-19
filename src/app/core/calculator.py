from src.app.core.api import Features


class Calculator:
    def calc_amount(
        self,
        proba: str,
        features: Features,
    ) -> int:
        """Функция принимает на вход вероятность дефолта и признаки и расчитывает одобренную сумму."""
        # Расчет базовой суммы займа в зависимости от пробы
        if proba < 0.1:
            base_amount = 500_000
        elif proba < 0.2:
            base_amount = 300_000
        else:
            base_amount = 100_000

        # Более высокий взвешенный скор внешних источников - более низкий риск дефолта
        if features.weighted_ext_score > 0.7:
            # максимальная сумма займа - 500_000
            base_amount = min(base_amount * 1.2, 500_000)
        elif features.weighted_ext_score < 0.3:
            base_amount = base_amount * 0.8

        # Если платеж составляет большую часть от дохода, то клиент менее надежный
        if features.annuity_to_income_proportion > 0.5:
            base_amount = base_amount * 0.8
        elif features.annuity_to_income_proportion < 0.2:
            base_amount = min(base_amount * 1.2, 500_000)

        # Изменение суммы займа в зависимости от стажа работы
        if features.days_employed < -2000:
            base_amount = min(base_amount * 1.2, 500_000)
        elif features.days_employed > -365:
            base_amount = base_amount * 0.7
        elif features.days_employed == 0:
            base_amount = base_amount * 0.5

        # Изменение суммы займа в зависимости от возраста (стат. тесты в прошлом задании показали значимость возраста клиента)
        age_years = abs(features.days_birth) / 365.2425
        if age_years < 25:
            base_amount = base_amount * 0.7
        elif age_years > 50:
            base_amount = min(base_amount * 1.05, 500_000)

        # Высокая ставка - высокий риск дефолта клиента
        if features.interest_rate > 20:
            base_amount = base_amount * 0.8

        # Минимальная сумма займа - 20_000
        base_amount = max(base_amount, 20_000)

        return int(round(base_amount, -2))
