import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, RandomApply, GaussianBlur, RandomAffine

from load_data import create_torch_dataset


def create_augmented_dataset():
    transform = Compose([
        RandomApply([GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5),
        RandomAffine(degrees=0, scale=(0.99, 1.03)),
    ])
    return create_torch_dataset(transform=transform)


# citation: https://braininformatics.springeropen.com/articles/10.1186/s40708-021-00144-2#Sec3
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
        self.dropout = nn.Dropout(p=0.1) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
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
    class_counts = np.bincount([label for _, label in dataset])
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
        num_samples=total_samples,
        replacement=True,
        drop_last = True # TODO: test false behavior
    )
    return sampler


def train_model_with_scheduler(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device="cpu"
):
    """
    Train a PyTorch model with a learning rate scheduler and validation tracking.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimizer (e.g., optim.Adam).
        scheduler: Learning rate scheduler.
        num_epochs: Number of epochs to train.
        device: Device to train on ('cuda' or 'cpu').

    Returns:
        train_loss_history, val_loss_history, val_accuracy_history
    """
    model.to(device)
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # TODO: consider removing

        # Update learning rate using the scheduler
        scheduler.step()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        # Print metrics for the epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    return train_loss_history, val_loss_history, val_accuracy_history


def plot_training_and_validation(train_loss, val_loss, val_accuracy, output_dir="figs", output_file="training_validation_plot.jpg"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.plot(val_loss, label="Val Loss", color="orange")
    plt.plot(val_accuracy, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training and validation metrics plot saved to: {output_path}")

def save_model_and_logs(
    model, optimizer, epoch, loss_history, output_dir="trained_models", fold_idx=None, metadata=None
):
    """
    Save the trained model checkpoint and a log file with training details for each fold.

    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): The number of training epochs.
        loss_history (list): List of loss values for each epoch.
        output_dir (str): Base directory to save the model and logs.
        fold_idx (int): Index of the current fold (for naming directories).
        metadata (dict): Additional metadata to log.
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
        "metadata": metadata  # Include metadata in the checkpoint
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # Save the log file with training details
    log_path = os.path.join(output_dir, "training_details.log")
    with open(log_path, "w") as log_file:
        log_file.write("Training Details\n")
        log_file.write("================\n")
        if fold_idx is not None:
            log_file.write(f"Fold Index: {fold_idx + 1}\n")
        log_file.write(f"Number of Epochs: {epoch}\n")
        log_file.write(f"Final Loss: {loss_history[-1]:.4f}\n")
        log_file.write("Loss History:\n")
        for i, loss in enumerate(loss_history, start=1):
            log_file.write(f"Epoch {i}: Loss = {loss:.4f}\n")
        log_file.write("\nMetadata:\n")
        if metadata:
            for key, value in metadata.items():
                log_file.write(f"{key}: {value}\n")
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
    model.eval() 
    all_labels = []
    all_predictions = []

    # Collect predictions and true labels
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
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
    balanced_accuracy = np.mean(class_accuracies)
    return balanced_accuracy


def stratified_cross_validation(dataset, n_splits=3, batch_size=2, num_workers=4, sample_validation=False):
    """
    Perform stratified 3-fold cross-validation with balanced sampling.

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
    i = 0
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        # Create Subsets for train and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create balanced samplers
        print("Train sampler for fold", i+1)
        train_sampler = create_balanced_sampler(train_subset)

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=num_workers
        )
        
        if sample_validation:
            print("Validation sampler for fold", i+1)
            val_sampler = create_balanced_sampler(val_subset)
            val_loader = DataLoader(
                dataset=val_subset,
                batch_size=batch_size,
                sampler=val_sampler,
                pin_memory=True,
                num_workers=num_workers
            )

        else:
            val_loader = DataLoader(
                dataset=val_subset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers
            )

        # Store loaders for this fold
        folds.append((train_loader, val_loader))

        i+=1

    return folds


def main(train_new_model=True, n_splits=3, sample_validation=True):
    print("Loading dataset...")

    # Load augmented dataset
    dataset = create_augmented_dataset()

    # Get stratified cross-validation folds with balanced sampling
    folds = stratified_cross_validation(
        dataset, 
        n_splits=n_splits, 
        batch_size=4, 
        num_workers=4, 
        sample_validation=sample_validation
    )

    # Cross-validation loop
    fold_metrics = []
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        # Define model, optimizer, and scheduler
        model = SimpleCNN3D(input_channels=1, num_classes=3)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if train_new_model:
            print("Training a new model...")
            criterion = nn.CrossEntropyLoss()
            num_epochs = 25

            # Train the model and track metrics
            train_loss, val_loss, val_accuracy = train_model_with_scheduler(
                model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device
            )

            # Plot training and validation metrics
            plot_training_and_validation(
                train_loss, val_loss, val_accuracy, output_dir=f"fold_{fold_idx + 1}_results"
            )

            # Save the trained model and logs
            save_model_and_logs(
                model, optimizer, num_epochs, train_loss,
                output_dir=f"trained_models/fold_{fold_idx + 1}",
                fold_idx=fold_idx,
                metadata={
                    "sample_validation": sample_validation, 
                    "n_splits": n_splits,
                    "num_epochs": num_epochs,
                    "optimizer": optimizer.__class__.__name__,
                    "lr_scheduler": scheduler.__class__.__name__,
                    "lr_scheduler_params": {"step_size": 45, "gamma": 0.1}
                }
            )
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
