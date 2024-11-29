import torch
import numpy as np
from torch.utils.data import DataLoader
from load_data import create_torch_dataset
from scipy.cluster.vq import kmeans, vq

def dataset_to_numpy(dataset, batch_size=32):
    """
    Converts a PyTorch Dataset into a single NumPy array of features.
    
    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset.
        batch_size (int): Batch size for loading data.

    Returns:
        np.ndarray: Array of shape (num_samples, feature_size).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    for img, _ in loader:  # Only extract the images, ignore labels
        all_features.append(img.view(img.size(0), -1).numpy())  # Flatten and convert to NumPy
    return np.vstack(all_features)  # Combine all batches into a single NumPy array

def apply_kmeans_scipy(features, n_clusters=10):
    """
    Applies K-Means clustering to the extracted features using scipy.

    Args:
        features (np.ndarray): Feature array of shape (num_samples, feature_size).
        n_clusters (int): Number of clusters.

    Returns:
        tuple: Cluster centroids and cluster assignments for each sample.
    """
    centroids, _ = kmeans(features, n_clusters)  # Compute K-Means centroids
    cluster_ids, _ = vq(features, centroids)  # Assign data points to the nearest cluster
    return centroids, cluster_ids

# Example Usage
if __name__ == "__main__":
    # Assume torch_dataset is already prepared (your NiftiDataset object)
    torch_dataset = create_torch_dataset()

    # Step 1: Convert PyTorch dataset to NumPy
    features = dataset_to_numpy(torch_dataset)
    print(f"Feature array shape: {features.shape}")  # Should be (num_samples, flattened_features)

    # Step 2: Apply K-Means using scipy
    centroids, cluster_assignments = apply_kmeans_scipy(features, n_clusters=10)
    print(f"Cluster centroids: {centroids}")
    print(f"Cluster assignments: {cluster_assignments}")
