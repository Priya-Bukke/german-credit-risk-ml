from src.data.prep_data import load_raw_data, prepare_features_and_target


def test_load_raw_data_shape():
    X, y = load_raw_data()
    assert X.shape[0] == 1000  # 1000 applicants
    assert X.shape[1] == 20    # 20 original attributes


def test_prepare_features_and_target():
    X, y = load_raw_data()
    X_encoded, y_binary = prepare_features_and_target(X, y)

    # Encoding should produce more columns (one-hot expands categories)
    assert X_encoded.shape[1] > X.shape[1]

    # Target should only contain 0s and 1s after binarization
    assert set(y_binary.unique()) == {0, 1}

    # Same number of rows in and out
    assert X_encoded.shape[0] == X.shape[0]
    assert y_binary.shape[0] == X.shape[0]