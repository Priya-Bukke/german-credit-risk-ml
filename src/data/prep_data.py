import pandas as pd
from ucimlrepo import fetch_ucirepo

COLUMN_NAMES = {
    "Attribute1": "checking_account_status",
    "Attribute2": "duration_months",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account",
    "Attribute7": "employment_years",
    "Attribute8": "installment_rate_pct",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_since_years",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "num_existing_credits",
    "Attribute17": "job",
    "Attribute18": "num_dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
}


def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the German Credit dataset from UCI and rename columns."""
    german_credit = fetch_ucirepo(id=144)
    X = german_credit.data.features.rename(columns=COLUMN_NAMES)
    y = german_credit.data.targets
    return X, y


def prepare_features_and_target(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categorical features and binarize the target (1 = bad risk)."""
    X_encoded = pd.get_dummies(X, drop_first=True)
    y_binary = (y["class"] == 2).astype(int)
    return X_encoded, y_binary


if __name__ == "__main__":
    X, y = load_raw_data()
    X_encoded, y_binary = prepare_features_and_target(X, y)
    print(f"Features: {X_encoded.shape}, Target: {y_binary.shape}")