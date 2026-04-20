from src.app.core.api import Features


class Calculator:
    max_amount = 300_000
    high_amount = 200_000
    medium_amount = 50_000
    min_amount = 20_000
    days_in_year = 365.2425

    def calc_amount(
        self,
        proba: float,
        features: Features,
    ) -> int:
        """Функция принимает на вход вероятность дефолта и признаки и расчитывает одобренную сумму."""
        amount = self._get_base_amount(proba)
        amount = self._apply_weighted_ext_score(amount, features.weighted_ext_score)
        amount = self._apply_annuity_to_income_proportion(
            amount, features.annuity_to_income_proportion
        )
        amount = self._apply_days_employed(amount, features.days_employed)
        amount = self._apply_days_birth(amount, features.days_birth)
        amount = self._apply_interest_rate(amount, features.interest_rate)
        amount = max(amount, self.min_amount)

        return int(round(amount, -2))

    def _get_base_amount(self, proba: float) -> float:
        """Расчет базовой суммы займа в зависимости от пробы."""
        if proba < 0.1:
            return self.max_amount
        elif proba < 0.2:
            return self.high_amount
        else:
            return self.medium_amount

    def _apply_weighted_ext_score(
        self, amount: float, weighted_ext_score: float
    ) -> float:
        """Более высокий взвешенный скор внешних источников - более низкий риск дефолта."""
        if weighted_ext_score > 0.7:
            # максимальная сумма займа - max_amount
            return min(amount * 1.5, self.max_amount)
        if weighted_ext_score < 0.5:
            return amount * 0.8
        return amount

    def _apply_annuity_to_income_proportion(
        self, amount: float, annuity_to_income_proportion: float
    ) -> float:
        """Если платеж составляет большую часть от дохода, то клиент менее надежный."""
        if annuity_to_income_proportion > 0.4:
            return amount * 0.8
        if annuity_to_income_proportion < 0.1:
            return min(amount * 1.2, self.max_amount)
        return amount

    def _apply_days_employed(self, amount: float, days_employed: float) -> float:
        """Изменение суммы займа в зависимости от стажа работы."""
        if days_employed < -3000:
            return min(amount * 1.2, self.max_amount)
        elif days_employed > -365:
            return amount * 0.7
        if days_employed == 0:
            return amount * 0.5
        return amount

    def _apply_days_birth(self, amount: float, days_birth: float) -> float:
        """
        Изменение суммы займа в зависимости от возраста.
        Стат. тесты в прошлом задании показали значимость возраста клиента.
        """
        age_years = abs(days_birth) / self.days_in_year
        if age_years < 25:
            return amount * 0.7
        if age_years > 50:
            return min(amount * 1.05, self.max_amount)
        return amount

    def _apply_interest_rate(self, amount: float, interest_rate: float) -> float:
        """Высокая ставка - высокий риск дефолта клиента."""
        if interest_rate > 20:
            return amount * 0.8
        return amount
