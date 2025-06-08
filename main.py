import numpy as np
import os
from data_loader import load_all_data, create_grid_data
from model import train_rf_model, predict_coordinates
from visualization import plot_results


def main():
    # Define data folder
    data_folder = "data"

    # Load all data from the data folder
    intensities, coordinates = load_all_data(data_folder)

    # Create 25x25 grid data
    X, y = create_grid_data(intensities, coordinates, grid_size=25)

    # Train Random Forest model
    rf_model, mse, X_test, y_test, y_pred = train_rf_model(X, y, n_estimators=100)

    # Print evaluation results
    print(f"Mean Squared Error: {mse:.6f}")

    # Visualize results
    plot_results(X_test, y_test, y_pred, mse)

    # Example prediction for a new intensity value
    # Example intensity with 8 features (2 per file: P_rec and P_rec_NLOS for 4 files)
    example_intensity = np.array([[0.2, 0.1, 0.18, 0.09, 0.15, 0.08, 0.12, 0.06]])
    predicted_coords = predict_coordinates(rf_model, example_intensity)
    print(
        f"Example prediction for intensity {example_intensity}: X={predicted_coords[0, 0]:.2f}, Y={predicted_coords[0, 1]:.2f}")


if __name__ == "__main__":
    main()