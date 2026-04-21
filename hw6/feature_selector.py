import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:
    """
    Класс для отбора признаков
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.importance_df = None

    def select_by_rf(self, X, y, top_n=50, n_estimators=100):
        """Отбор по Random Forest Importance"""

        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )

        self.selected_features = rf_importance.head(top_n).index.tolist()

        self.importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": rf.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"Отобрано {len(self.selected_features)} признаков по Random Forest")
        return X[self.selected_features]

    def get_selected_features(self):
        return self.selected_features

    def get_importance_df(self):
        return self.importance_df
