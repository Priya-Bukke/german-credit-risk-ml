from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, params: dict):
    if name == "logistic_regression":
        estimator = LogisticRegression(**params, class_weight="balanced", random_state=42)
    elif name == "random_forest":
        estimator = RandomForestClassifier(**params, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return make_pipeline(StandardScaler(), estimator)