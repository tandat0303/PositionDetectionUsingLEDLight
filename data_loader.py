import numpy as np
import pandas as pd
import os


def load_all_data(data_folder):
    """
    Load all P_rec, P_rec_NLOS, XR, and YR files from the data folder and combine them.
    Returns combined intensities and coordinates.
    """
    # Initialize lists to store intensities
    all_p_rec = []
    all_p_rec_nlos = []

    # Define file patterns
    p_rec_files = [f for f in os.listdir(data_folder) if f.startswith('P_rec_A') and f.endswith('.csv')]
    p_rec_nlos_files = [f for f in os.listdir(data_folder) if f.startswith('P_rec_NLOS_A') and f.endswith('.csv')]

    # Load all P_rec files
    for file in p_rec_files:
        p_rec = pd.read_csv(os.path.join(data_folder, file), header=None).values
        all_p_rec.append(p_rec)

    # Load all P_rec_NLOS files
    for file in p_rec_nlos_files:
        p_rec_nlos = pd.read_csv(os.path.join(data_folder, file), header=None).values
        all_p_rec_nlos.append(p_rec_nlos)

    # Stack all P_rec and P_rec_NLOS matrices
    p_rec_combined = np.stack(all_p_rec, axis=-1)
    p_rec_nlos_combined = np.stack(all_p_rec_nlos, axis=-1)

    # Ensure all matrices have the same shape
    assert p_rec_combined.shape == p_rec_nlos_combined.shape, "P_rec and P_rec_NLOS matrices must have the same shape"

    # Load XR and YR files
    xr_files = [f for f in os.listdir(data_folder) if f == 'XR.csv']
    yr_files = [f for f in os.listdir(data_folder) if f == 'YR.csv']
    assert len(xr_files) == 1 and len(yr_files) == 1, "Exactly one XR.csv and one YR.csv file must exist"
    xr = pd.read_csv(os.path.join(data_folder, xr_files[0]), header=None).values
    yr = pd.read_csv(os.path.join(data_folder, yr_files[0]), header=None).values

    # Ensure XR and YR have the same shape as P_rec and P_rec_NLOS
    assert p_rec_combined.shape[0] == xr.shape[0] and p_rec_combined.shape[1] == xr.shape[1], \
        "P_rec and XR/YR matrices must have compatible shapes"
    assert xr.shape == yr.shape, "XR and YR matrices must have the same shape"

    # Reshape and combine intensities
    intensities = np.stack([p_rec_combined, p_rec_nlos_combined], axis=-1).reshape(-1, 2 * len(p_rec_files))

    # Flatten X and Y coordinates
    x_coords = xr.flatten()
    y_coords = yr.flatten()

    # Create coordinate pairs
    coordinates = np.stack([x_coords, y_coords], axis=-1)

    return intensities, coordinates


def create_grid_data(intensities, coordinates, grid_size=25):
    """
    Reshape data for a grid of specified size (e.g., 25x25).
    Returns data suitable for training.
    """
    # Reshape intensities to grid_size x grid_size x (2 * number_of_files)
    grid_intensities = intensities.reshape(grid_size, grid_size, -1)
    grid_coordinates = coordinates.reshape(grid_size, grid_size, 2)

    # Flatten for model input
    X = grid_intensities.reshape(-1, grid_intensities.shape[-1])
    y = grid_coordinates.reshape(-1, 2)

    return X, y