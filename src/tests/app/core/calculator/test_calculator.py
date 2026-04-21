import pytest
from unittest.mock import Mock

from src.app.core.api import Features
from src.app.core.calculator import Calculator


@pytest.fixture
def calc():
    """Фикстура калькулятора для всех тестов."""
    return Calculator()


@pytest.mark.parametrize(
    "proba,expected",
    [
        pytest.param(0.0, Calculator.max_amount, id="proba_0.0_max_amount"),
        pytest.param(0.09, Calculator.max_amount, id="proba_0.09_max_amount"),
        pytest.param(0.1, Calculator.high_amount, id="proba_0.1_high_amount"),
        pytest.param(0.19, Calculator.high_amount, id="proba_0.19_high_amount"),
        pytest.param(0.2, Calculator.medium_amount, id="proba_0.2_medium_amount"),
        pytest.param(0.3, Calculator.medium_amount, id="proba_0.3_medium_amount"),
    ],
)
def test_get_base_amount(calc: Calculator, proba: float, expected: int):
    """Тест: базовая сумма в зависимости от вероятности."""
    assert calc._get_base_amount(proba) == expected


# Методы для изменения суммы однотипны, можно объединить тесты в один общий
# По одному тест-кейсу на каждый интервал, базовая сумма = 100_000 для каждого теста
@pytest.mark.parametrize(
    "method_name,value,expected",
    [
        # _apply_weighted_ext_score
        pytest.param("_apply_weighted_ext_score", 0.9, 150_000, id="score_increase"),
        pytest.param("_apply_weighted_ext_score", 0.7, 100_000, id="score_no_change"),
        pytest.param("_apply_weighted_ext_score", 0.49, 80_000, id="score_decrease"),
        # _apply_annuity_to_income_proportion
        pytest.param(
            "_apply_annuity_to_income_proportion", 0.9, 80_000, id="proportion_decrease"
        ),
        pytest.param(
            "_apply_annuity_to_income_proportion",
            0.4,
            100_000,
            id="proportion_no_change",
        ),
        pytest.param(
            "_apply_annuity_to_income_proportion",
            0.09,
            120_000,
            id="proportion_increase",
        ),
        # _apply_days_employed
        pytest.param("_apply_days_employed", -4000, 120_000, id="days_increase"),
        pytest.param("_apply_days_employed", -3000, 100_000, id="days_no_change"),
        pytest.param("_apply_days_employed", -364, 70_000, id="days_decrease_30"),
        pytest.param("_apply_days_employed", 0, 50_000, id="days_decrease_50"),
        # _apply_days_birth
        pytest.param("_apply_days_birth", -8000, 70_000, id="age_young"),
        pytest.param("_apply_days_birth", -15000, 100_000, id="age_adult"),
        pytest.param("_apply_days_birth", -19000, 105_000, id="age_senior"),
        # _apply_interest_rate
        pytest.param("_apply_interest_rate", 25, 80_000, id="rate_decrease"),
        pytest.param("_apply_interest_rate", 20, 100_000, id="rate_no_change"),
    ],
)
def test_coef_methods(calc: Calculator, method_name: str, value: float, expected: int):
    """Общий тест для всех методов-модификаторов суммы займа с помощью коэффициентов."""
    amount = 100_000
    method = getattr(calc, method_name)
    assert method(amount, value) == expected


@pytest.mark.parametrize(
    "proba,weighted_score,annuity_proportion,days_employed,days_birth,interest_rate,expected",
    [
        # Все факторы положительные, amount = максимальная сумма
        pytest.param(
            0.05,
            0.9,
            0.05,
            -4000,
            -20000,
            10,
            Calculator.max_amount,
            id="all_positive_factors_max",
        ),
        # Все факторы отрицательные, amount = минимальная сумма
        pytest.param(
            0.3,
            0.4,
            0.5,
            0,
            -5000,
            25,
            Calculator.min_amount,
            id="all_negative_factors_min",
        ),
        # Все факторы нейтральные, amount = базовая сумма high_amount
        pytest.param(
            0.15,
            0.6,
            0.2,
            -1000,
            -15000,
            15,
            Calculator.high_amount,
            id="neutral_factors_high_amount",
        ),
    ],
)
def test_calc_amount_integration(
    calc: Calculator,
    proba: float,
    weighted_score: float,
    annuity_proportion: float,
    days_employed: int,
    days_birth: int,
    interest_rate: float,
    expected: int,
):
    """Тест calc_amount."""
    #  с помощью Mock-обекта Features
    features = Mock(spec=Features)
    features.weighted_ext_score = weighted_score
    features.annuity_to_income_proportion = annuity_proportion
    features.days_employed = days_employed
    features.days_birth = days_birth
    features.interest_rate = interest_rate

    result = calc.calc_amount(proba, features)

    assert result == expected
    assert isinstance(result, int)
    assert result % 100 == 0
    assert calc.min_amount <= result <= calc.max_amount
