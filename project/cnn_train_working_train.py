import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support

from load_data import create_torch_dataset  # Assuming this returns a torch Dataset


class SimpleCNN3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(SimpleCNN3D, self).__init__()

        # Define 5 convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Flatten size calculation: manually compute the final dimensions
        self.flatten_size = 128 * 2 * 2 * 2

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.elu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout here
        x = self.fc2(x)
        return x


def create_balanced_sampler(dataset):
    """
    Create a WeightedRandomSampler for balanced sampling.

    Args:
        dataset: PyTorch dataset object with (input, label) pairs.

    Returns:
        sampler: WeightedRandomSampler for balanced sampling.
    """
    # Step 1: Count the frequency of each class
    class_counts = np.bincount([label for _, label in dataset])  # Replace with actual dataset labels
    total_samples = len(dataset)
    print(f"Class Counts: {class_counts}")

    # Step 2: Calculate weights for each class
    class_weights = 1.0 / class_counts
    print(f"Class Weights: {class_weights}")

    # Step 3: Assign a weight to each sample in the dataset
    sample_weights = [class_weights[label] for _, label in dataset]

    # Step 4: Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,  # Total samples to draw in each epoch
        replacement=True           # Samples can be drawn multiple times per epoch
    )
    return sampler


# Training loop with tqdm
def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device="cpu"):
    model.to(device)
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Create a tqdm progress bar for the batches in this epoch
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                running_loss += loss.item() * inputs.size(0)

                # Update tqdm with the current loss
                pbar.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return train_loss_history


def plot_loss(loss_history, output_dir="figs", output_file="training_loss.jpg"):
    """
    Plot training loss and save it as a .jpg file.

    Args:
        loss_history (list): List of loss values for each epoch.
        output_dir (str): Directory to save the output file.
        output_file (str): Name of the output file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free up memory
    print(f"Training loss plot saved to: {output_path}")


def save_model_and_logs(model, optimizer, epoch, loss_history, output_dir="trained_models", fold_idx=None):
    """
    Save the trained model checkpoint and a log file with training details for each fold.

    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): The number of training epochs.
        loss_history (list): List of loss values for each epoch.
        output_dir (str): Base directory to save the model and logs.
        fold_idx (int): Index of the current fold (for naming directories).
    """
    # Create a fold-specific directory
    if fold_idx is not None:
        output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the model checkpoint
    checkpoint_path = os.path.join(output_dir, "model_checkpoint.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # Save the log file with training details
    log_path = os.path.join(output_dir, "training_details.log")
    with open(log_path, "w") as log_file:
        log_file.write("Training Details\n")
        log_file.write("================\n")
        log_file.write(f"Fold Index: {fold_idx + 1}\n")
        log_file.write(f"Number of Epochs: {epoch}\n")
        log_file.write(f"Final Loss: {loss_history[-1]:.4f}\n")
        log_file.write("Loss History:\n")
        for i, loss in enumerate(loss_history, start=1):
            log_file.write(f"Epoch {i}: Loss = {loss:.4f}\n")
    print(f"Training details log saved to: {log_path}")


def evaluate_model_on_test_fold(model, data_loader, device, fold_idx):
    """
    Evaluate the model on the validation/test fold and save metrics for the fold.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for the validation/test fold.
        device: The device ('cuda' or 'cpu') on which the model is running.
        fold_idx: Current fold index (for saving results with fold-specific names).
    """
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_predictions = []

    # Collect predictions and true labels
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(3)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Save confusion matrix plot
    output_dir = f"fold_{fold_idx + 1}_results"
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"confusion_matrix_fold_{fold_idx + 1}.jpg")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Class 0", "Class 1", "Class 2"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Fold {fold_idx + 1})")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Save classification report
    report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2"], digits=2)
    report_path = os.path.join(output_dir, f"classification_report_fold_{fold_idx + 1}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # Calculate metrics to return
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    balanced_accuracy = np.mean(class_accuracies)  # Balanced accuracy (mean recall)
    return balanced_accuracy


def stratified_cross_validation(dataset, n_splits=3, batch_size=2, num_workers=4):
    """
    Perform stratified 3-fold cross-validation with PyTorch.

    Args:
        dataset: PyTorch dataset with (input, label) pairs.
        n_splits: Number of folds for cross-validation (default: 3).
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.

    Returns:
        List of (train_loader, val_loader) tuples for each fold.
    """
    # Extract labels for stratification
    labels = [label for _, label in dataset]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create DataLoaders for each fold
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        # Create Subsets for train and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle within training fold
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=num_workers,
            pin_memory=True
        )

        # Store loaders for this fold
        folds.append((train_loader, val_loader))

    return folds


def main(train_new_model=True, n_splits=3):
    print("Loading dataset...")

    # Load dataset
    dataset = create_torch_dataset()
    labels = [label for _, label in dataset]  # Extract labels for stratification

    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation loop
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create a balanced sampler for the training subset
        train_sampler = create_balanced_sampler(train_subset)

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=2, 
            sampler=train_sampler,  # Use balanced sampling for training
            pin_memory=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=2, 
            shuffle=False,  # No need to shuffle validation data
            pin_memory=True, 
            num_workers=4
        )

        # Define model and optimizer
        model = SimpleCNN3D(input_channels=1, num_classes=3)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if train_new_model:
            print("Training a new model...")
            criterion = nn.CrossEntropyLoss()
            num_epochs = 100
            loss_history = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
            plot_loss(loss_history, output_dir=f"fold_{fold_idx + 1}_results", output_file="training_loss.jpg")

            # Save the trained model and logs for the current fold
            save_model_and_logs(model, optimizer, num_epochs, loss_history, output_dir="trained_models", fold_idx=fold_idx)
        else:
            print("Pre-trained model loading is not implemented yet for cross-validation.")

        # Evaluate the model on the validation fold
        print("Evaluating the model...")
        fold_accuracy = evaluate_model_on_test_fold(model, val_loader, device, fold_idx)
        fold_metrics.append(fold_accuracy)
        print(f"Fold {fold_idx + 1} Balanced Accuracy: {fold_accuracy:.2f}")

    # Summary of cross-validation results
    print("\nCross-validation complete.")
    print(f"Average Balanced Accuracy Across Folds: {np.mean(fold_metrics):.2f}")

if __name__ == "__main__":
    main()