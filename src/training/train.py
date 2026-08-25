import mlflow
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data.prep_data import load_raw_data, prepare_features_and_target
from src.training.model_factory import build_model
from src.utils.config import load_config


def suggest_params(trial, model_name: str, search_space: dict) -> dict:
    params = {}
    for param_name, bounds in search_space.items():
        low, high = bounds
        params[param_name] = trial.suggest_int(param_name, low, high)
    return params


def train_model():
    config = load_config()
    mlflow.set_experiment("german-credit-risk")

    X, y = load_raw_data()
    X_encoded, y_binary = prepare_features_and_target(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_binary,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y_binary,
    )

    best_overall = {"name": None, "score": -1, "params": None}

    for model_config in config["models"]:
        name = model_config["name"]
        search_space = model_config["search_space"]

        def objective(trial):
            params = suggest_params(trial, name, search_space)
            model = build_model(name, params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config["optuna"]["n_trials"], show_progress_bar=False)

        print(f"\n=== {name}: best CV F1 = {study.best_value:.4f} ===")
        print("Best params:", study.best_params)

        # Log this model's best result as one MLflow run
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_params(study.best_params)
            mlflow.log_metric("cv_f1", study.best_value)

            # Fit this model's best version and log test-set metrics too
            candidate_model = build_model(name, study.best_params)
            candidate_model.fit(X_train, y_train)
            y_pred = candidate_model.predict(X_test)

            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

        if study.best_value > best_overall["score"]:
            best_overall = {"name": name, "score": study.best_value, "params": study.best_params}

    print(f"\n=== Overall best: {best_overall['name']} ===")
    print(f"CV F1: {best_overall['score']:.4f}")
    print(f"Params: {best_overall['params']}")

    final_model = build_model(best_overall["name"], best_overall["params"])
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    print(f"\n=== Final test set performance ({best_overall['name']}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_model()