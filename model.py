from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def train_rf_model(X, y, n_estimators=100, test_size=0.2, random_state=42):
    """
    Train a Random Forest model to predict coordinates from intensities.
    Returns trained model and evaluation metrics.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return rf, mse, X_test, y_test, y_pred


def predict_coordinates(model, intensities):
    """
    Predict coordinates for given intensity values.
    """
    return model.predict(intensities)