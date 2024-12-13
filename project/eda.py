import os
import torch
import numpy as np
from load_data import create_torch_dataset
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN


def make_slice_grid_plot(img, axis, output_file="figs/slice_grid.png"):
    """
    Generate a 5x5 grid of slices from a 3D dataset and save it as an image.

    Args:
        img (torch.Tensor): 4D image data (channels, depth, height, width).
        axis (int): Axis along which to take the slices (0, 1, or 2).
        output_file (str): Path to save the grid image.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Remove channel dimension for 3D slicing
    img = img.squeeze(0)

    # Determine slice indices (equally spaced)
    num_slices = 25
    max_index = img.shape[axis]
    indices = np.linspace(0, max_index - 1, num_slices, dtype=int)

    # Create a 5x5 grid for plotting
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f"5x5 Slice Grid (Axis {axis})", fontsize=16)

    for ax, idx in zip(axes.ravel(), indices):
        # Extract the slice based on the axis
        if axis == 0:
            slice_data = img[idx, :, :].numpy()
        elif axis == 1:
            slice_data = img[:, idx, :].numpy()
        else:
            slice_data = img[:, :, idx].numpy()

        # Plot the slice
        ax.imshow(slice_data, cmap="gray")
        ax.set_title(f"Slice {idx}", fontsize=8)
        ax.axis("off")

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Slice grid saved to {output_file}")


def three_dim_view(img, output_dir="figs", view_interior=False):
    os.makedirs(output_dir, exist_ok=True)

    # Remove channel dimension for 3D processing
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).numpy()

    if view_interior:
        # Calculate midpoints along each dimension
        cut_x = img.shape[2] // 2
        cut_y = img.shape[1] // 2
        cut_z = img.shape[0] // 2

        # Remove one corner cube (e.g., top-front-left)
        img[:cut_z, :cut_y, cut_x:] = 0

    # Get the indices of non-zero voxels
    z, y, x = np.nonzero(img)
    values = img[z, y, x]

    # Filter for high-intensity regions (optional)
    mask = values > 0.1  # Adjust threshold as needed
    x, y, z, values = x[mask], y[mask], z[mask], values[mask]

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        x=x, y=y, z=z, color=values, title="3D Scatter Plot of Non-Zero Voxels",
        labels={"x": "Width", "y": "Height", "z": "Depth"},
        color_continuous_scale="Viridis"
    )

    # Reduce opacity for better visualization
    fig.update_traces(marker=dict(opacity=1))

    # Save the plot
    output_file = os.path.join(output_dir, "figure.html")
    fig.write_html(output_file)
    print(f"HTML saved to: {output_file}. Open it in your browser to interact with the plot!")


def cluster_slice(img_slice, eps=0.1, min_samples=10):
    """
    Perform DBSCAN clustering on a specific 2D slice of an image.

    Args:
        img_slice (numpy.ndarray): The 2D image slice to cluster.
        eps (float): The DBSCAN neighborhood size (radius of a neighborhood).
        min_samples (int): The minimum number of points required to form a cluster.

    Returns:
        numpy.ndarray: Cluster labels for the slice, reshaped to the original slice shape.
    """

    # Use intensity only
    features = img_slice.reshape(-1, 1)

    # Perform DBSCAN clustering
    scanner = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = scanner.fit_predict(features)

    # Print summary of clusters
    unique_clusters = np.unique(cluster_labels)
    print(f"Number of clusters (excluding noise): {len(unique_clusters[unique_clusters != -1])}")
    print(f"Noise points: {np.sum(cluster_labels == -1)}")

    # Reshape the cluster labels back to the original slice shape
    return cluster_labels.reshape(img_slice.shape)


def plot_clusters_with_option(img_slice, cluster_labels, show_original=False):
    """
    Plot the cluster labels, with an option to also display the original slice side by side.

    Args:
        img_slice (numpy.ndarray): The original image slice (2D array).
        cluster_labels (numpy.ndarray): The cluster labels for the slice (2D array).
        show_original (bool): If True, plot both the original slice and cluster labels side by side.
                              If False, plot only the cluster labels.
    """
    if show_original:
        # Create a 2x1 grid for the plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the original slice
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("Original Slice")
        axes[0].set_xlabel("Y-axis")
        axes[0].set_ylabel("Z-axis")

        # Plot the clustered slice
        axes[1].imshow(cluster_labels, cmap="tab20")  # Use a discrete colormap
        axes[1].set_title("Clustered Slice")
        axes[1].set_xlabel("Y-axis")

        # Add colorbar for the clustered slice
        cbar = fig.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)
        cbar.set_label("Cluster Label")

        # Adjust layout and display the plots
        plt.tight_layout()

    else:
        # Plot only the clustered slice
        plt.figure(figsize=(8, 6))
        plt.imshow(cluster_labels, cmap="tab20")  # Use a discrete colormap
        plt.colorbar(label="Cluster Label")
        plt.title("Clustered Slice")
        plt.xlabel("Y-axis")
        plt.ylabel("Z-axis")

    # Save the plot
    output_dir = "figs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "clustered.jpg")
    plt.savefig(output_file, dpi=300)
    print(f"JPG saved to: {output_file}.")


def main():
    # Decide which EDA to run
    make_slice_grid_plot_ = True
    three_dim_view_ = True
    k_means_ = True

    # Load your dataset
    dataset = create_torch_dataset()

    # Generate a 5x5 grid of slices along the first axis
    if make_slice_grid_plot_:
        img, _ = dataset[0]
        for i in range(3):
            make_slice_grid_plot(img, axis=int(i), output_file=f"figs/slice_grid_axis{int(i)}.png")

    # Generate a 3D visualization
    if three_dim_view_:
        img, _ = dataset[0]
        three_dim_view(img, view_interior=True)

    # Implementation of K-means (style) algorithm for clustering
    if k_means_:
        img, _ = dataset[0]
        img_slice = img.squeeze(0).numpy()[22, :, :]
        cluster_labels = cluster_slice(img_slice)
        plot_clusters_with_option(img_slice, cluster_labels, True)


if __name__ == "__main__":
    main()
