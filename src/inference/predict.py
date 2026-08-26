import joblib
import pandas as pd

from src.utils.paths import PROJECT_ROOT


def load_model():
    models_dir = PROJECT_ROOT / "models"
    model = joblib.load(models_dir / "champion_model.pkl")
    feature_columns = joblib.load(models_dir / "feature_columns.pkl")
    return model, feature_columns


def predict(raw_applicant: dict) -> dict:
    """
    Take a dict of raw applicant fields (same shape as the original dataset columns,
    e.g. checking_account_status, duration_months, ...) and return a prediction.
    """
    model, feature_columns = load_model()

    # Turn the single applicant into a one-row DataFrame
    df = pd.DataFrame([raw_applicant])

    # One-hot encode the same way training data was encoded
    df_encoded = pd.get_dummies(df)

    # Align columns: add any missing dummy columns as 0, drop any extras, keep training order
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df_aligned)[0]
    probability = model.predict_proba(df_aligned)[0][1]  # probability of class 1 (bad risk)

    return {
        "prediction": "bad_risk" if prediction == 1 else "good_risk",
        "bad_risk_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    # Example applicant, for a quick manual test
    example_applicant = {
        "checking_account_status": "A11",
        "duration_months": 24,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 3500,
        "savings_account": "A61",
        "employment_years": "A73",
        "installment_rate_pct": 3,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since_years": 2,
        "property": "A121",
        "age": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "num_existing_credits": 1,
        "job": "A173",
        "num_dependents": 1,
        "telephone": "A192",
        "foreign_worker": "A201",
    }

    result = predict(example_applicant)
    print(result)