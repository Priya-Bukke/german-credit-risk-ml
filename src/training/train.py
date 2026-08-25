from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.data.prep_data import load_raw_data, prepare_features_and_target
from src.utils.config import load_config


def train_model():
    config = load_config()

    # Load and prepare data
    X, y = load_raw_data()
    X_encoded, y_binary = prepare_features_and_target(X, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_binary,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y_binary,
    )

    # Train
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=config["model"]["max_iter"]),
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_model()