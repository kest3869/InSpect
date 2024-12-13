import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def generate_labels(top_dir, subsets, labels, use_unscaled):
    """
    Generates labeled dataset paths.

    Args:
        top_dir (str): Path to the dataset directory.
        subsets (list): List of subsets (e.g., 'healthy', 'ptsd', 'pure_mdd').
        labels (dict): Mapping of subset names to labels.
        use_unscaled (bool): Whether to use unscaled files.

    Returns:
        np.array: Array where each entry is [file_path, label].
    """
    dataset = []
    for subset in subsets:
        base_path = os.path.join(top_dir, subset)
        for file in os.listdir(base_path):
            if (file.endswith('.nii') or file.endswith('.nii.gz')) and (('max' not in file) if use_unscaled else ('max' in file)):
                dataset.append([os.path.join(base_path, file), labels[subset]])
    return np.array(dataset)


class NiftiDataset(Dataset):
    """
    A custom PyTorch Dataset for NIfTI files.
    """
    def __init__(self, dataset_array, transform=None):
        """
        Args:
            dataset_array (np.array): Array where each entry is [file_path, label].
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.dataset = dataset_array
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        file_path, label = self.dataset[idx]
        
        # Load NIfTI file
        img = nib.load(file_path).get_fdata()

        # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Adds channel dimension to make [1, depth, height, width]

        # Convert NumPy array to PyTorch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Apply transform if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)  # Transforms expect PyTorch tensors now

        # Convert label to tensor
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return img_tensor, label_tensor

    ''' # used for cnn_train_working.py
    def __getitem__(self, idx):
        file_path, label = self.dataset[idx]
        
        # Load NIfTI file
        img = nib.load(file_path).get_fdata()

        # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Adds channel dimension to make [1, depth, height, width]

        # Apply transform if provided
        if self.transform:
            img = self.transform(img)

        # Convert to PyTorch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)  # Shape: [1, depth, height, width]
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return img_tensor, label_tensor
    '''


def generate_dataset(dataset_array, transform=None):
    """
    Prepares a PyTorch Dataset object from the given np.array dataset.

    Args:
        dataset_array (np.array): Array where each entry is [file_path, label].
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        NiftiDataset: A PyTorch Dataset object.
    """
    return NiftiDataset(dataset_array, transform)


def create_torch_dataset(
    top_dir='/home/InSpect/data/datasets',
    subsets=None,
    labels=None,
    use_unscaled=False,
    transform=None
):
    """
    End-to-end function to create a PyTorch dataset.

    Args:
        top_dir (str, optional): Path to the dataset directory. Default is '/home/InSpect/data/datasets'.
        subsets (list, optional): List of subsets (e.g., 'healthy', 'ptsd', 'pure_mdd'). Default is ['healthy', 'ptsd', 'pure_mdd'].
        labels (dict, optional): Mapping of subset names to labels. Default is {'healthy': '0', 'ptsd': '1', 'pure_mdd': '2'}.
        use_unscaled (bool, optional): Whether to use unscaled files. Default is False (use max-scaled).
        transform (callable, optional): Optional transform to apply to the data. Default is None.

    Returns:
        NiftiDataset: A PyTorch Dataset object.
    """
    if subsets is None:
        subsets = ['healthy', 'ptsd', 'pure_mdd']
    if labels is None:
        labels = {'healthy': '0', 'ptsd': '1', 'pure_mdd': '2'}

    dataset_array = generate_labels(top_dir, subsets, labels, use_unscaled)
    torch_dataset = generate_dataset(dataset_array, transform)
    return torch_dataset



def main():
    top_dir = '/home/InSpect/data/datasets'
    subsets = ['healthy', 'ptsd', 'pure_mdd']
    labels = {'healthy': '0', 'ptsd': '1', 'pure_mdd': '2'}

    # Create a PyTorch Dataset
    use_unscaled = False  # Set based on your preference or pass as an argument
    torch_dataset = create_torch_dataset(top_dir, subsets, labels, use_unscaled)

    # Example Usage
    img, label = torch_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")

    # Extract a slice (e.g., middle slice along the first axis)
    slice_index = img.shape[0] // 2
    image_slice = img[slice_index, :, :]

    # Create 'figs' directory if it doesn't exist
    os.makedirs("figs", exist_ok=True)

    # Plot and save the slice
    plt.imshow(image_slice.numpy(), cmap='gray')
    plt.title(f"Slice {slice_index}")
    plt.colorbar()
    output_path = os.path.join("figs", f"slice_{slice_index}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
