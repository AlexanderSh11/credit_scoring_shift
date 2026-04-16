from pathlib import Path
import pickle
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .data_loading import save_dataframe
from .data_preprocessing import scale_features


class ModelTrainer:
    """
    Класс для обучения моделей
    """

    def __init__(self, random_state=42, models_dir="models", data_path_dir=None):
        self.random_state = random_state
        self.best_params = {}
        self.results = {}
        self.models = {}
        self.models_dir = Path(models_dir)
        self.data_path_dir = data_path_dir

    def save_model(self, model, model_name):
        """Сохранение модели в pickle файл"""
        filename = self.models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Модель сохранена: {filename}")

    def get_selected_data(self, X_train, X_test, selector, top_n):
        """Отбор признаков"""
        selected_features = selector.get_selected_features()[:top_n]
        return X_train[selected_features], X_test[selected_features]

    def scale(self, X_train, X_test):
        """Масштабирование для моделей, которым это нужно"""
        X_train_scaled, scaler = scale_features(X_train, X_train.columns.tolist())
        X_test_scaled = X_test.copy()
        X_test_scaled[X_train.columns] = scaler.transform(X_test[X_train.columns])
        return X_train_scaled, X_test_scaled

    def print_metrics(self, y_test, y_pred, y_pred_proba, model_name):
        """Вывод метрик"""
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Метрики {model_name}")
        print(f"ROC-AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return auc

    def train_model(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        selector,
        top_n,
        need_scale,
        model_name,
        param_grid=None,
        data_filename=None,
    ):
        """Обучение модели"""
        print(model_name)
        X_train_sel, X_test_sel = self.get_selected_data(
            X_train, X_test, selector, top_n
        )
        print(f"Признаков: {X_train_sel.shape[1]}")

        # Масштабирование (если нужно)
        if need_scale:
            X_train_sel, X_test_sel = self.scale(X_train_sel, X_test_sel)

        if self.data_path_dir and data_filename:
            save_dataframe(
                df=X_train_sel,
                path_dir=self.data_path_dir,
                filename=f"{data_filename}_train.csv",
            )
            save_dataframe(
                df=X_test_sel,
                path_dir=self.data_path_dir,
                filename=f"{data_filename}_test.csv",
            )
            save_dataframe(
                df=pd.DataFrame(y_train),
                path_dir=self.data_path_dir,
                filename=f"{data_filename}_y_train.csv",
            )
            save_dataframe(
                df=pd.DataFrame(y_test),
                path_dir=self.data_path_dir,
                filename=f"{data_filename}_y_test.csv",
            )

        start_time = time.time()

        if param_grid:
            print(f"Подбор гиперпараметров для {model_name}")
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_sel, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            print(f"Лучшие параметры: {grid_search.best_params_}")
        else:
            model.fit(X_train_sel, y_train)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Время обучения: {elapsed:.2f} секунд")

        y_pred = model.predict(X_test_sel)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]

        auc = self.print_metrics(y_test, y_pred, y_pred_proba, model_name)

        self.models[model_name] = model
        self.results[model_name] = auc

        self.save_model(model, model_name)

        return model, auc

    def train_logistic_regression(
        self, X_train, X_test, y_train, y_test, selector, top_n=20
    ):
        """Логистическая регрессия"""
        model = LogisticRegression(
            random_state=self.random_state, max_iter=1000, class_weight="balanced"
        )

        param_grid = {
            "C": [0.01, 0.1, 1],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }

        return self.train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            selector,
            top_n,
            need_scale=True,
            model_name="Logistic Regression",
            param_grid=param_grid,
            data_filename="scaled_data",
        )

    def train_decision_tree(self, X_train, X_test, y_train, y_test, selector, top_n=20):
        """Дерево решений"""
        model = DecisionTreeClassifier(
            random_state=self.random_state, class_weight="balanced"
        )

        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4],
        }

        return self.train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            selector,
            top_n,
            need_scale=False,
            model_name="Decision Tree",
            param_grid=param_grid,
        )

    def train_random_forest(self, X_train, X_test, y_train, y_test, selector, top_n=20):
        """Случайный лес"""
        model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, class_weight="balanced"
        )

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5, 10],
        }

        return self.train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            selector,
            top_n,
            need_scale=False,
            model_name="Random Forest",
            param_grid=param_grid,
        )

    def train_gradient_boosting(
        self, X_train, X_test, y_train, y_test, selector, top_n=20
    ):
        """Градиентный бустинг"""
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state,
            learning_rate=0.1,
            max_depth=3,
        )

        param_grid = {
            "n_estimators": [200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 7],
        }

        return self.train_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            selector,
            top_n,
            need_scale=False,
            model_name="Gradient Boosting",
            param_grid=param_grid,
        )

    def train_all(self, X_train, X_test, y_train, y_test, selector, top_n=20):
        """Обучение всех моделей"""
        self.train_logistic_regression(
            X_train, X_test, y_train, y_test, selector, top_n
        )
        self.train_decision_tree(X_train, X_test, y_train, y_test, selector, top_n)
        self.train_random_forest(X_train, X_test, y_train, y_test, selector, top_n)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test, selector, top_n)

        return self.results
