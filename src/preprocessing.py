import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_features_target(df, target_col='cardio'):
    """
    Split dataframe into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_data(X_train, X_test, save_scaler=False, scaler_path='models/scaler.pkl'):
    """
    Scale features using StandardScaler.
    Fit only on training data to prevent data leakage.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, scaler
