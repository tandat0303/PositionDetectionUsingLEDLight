import matplotlib.pyplot as plt
import numpy as np


def plot_results(X_test, y_test, y_pred, mse):
    """
    Visualize true vs predicted coordinates and display MSE.
    """
    plt.figure(figsize=(10, 8))

    # Plot true coordinates
    plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='True Coordinates', alpha=0.6)

    # Plot predicted coordinates
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Predicted Coordinates', alpha=0.6)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'True vs Predicted Coordinates (MSE: {mse:.6f})')
    plt.legend()
    plt.grid(True)
    plt.show()