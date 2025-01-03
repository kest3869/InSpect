import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


def main():

    # Define Hyperparameters
    hyperparameters = {
        'n_features': [65, 900, 250], # training and validation
        'n_features_test' : [3000, 3000, 3000], # testing
        'distributions': [
            [-0.15, 0, 0.2],  # means
            [1.6, 2.0, 1.75]  # standard deviations
        ],
        'learning_rate': 0.001, # ADAM
        'weight_decay': 0.00005, # ADAM
        'epochs': 100,
        'use_train_sampler': True,
        'use_val_sampler': False,
        'use_weighted_loss_train' : False,
        'alpha' : 1.00, # weighted cross-entropy
        'beta' : 1.0, # random sampler
        'eta_min': 0.00001, # minimum learning rate
        'data_augmentation' : False,
        'output_dir': './tests/scrap/',
    }

    # Prepare datasets and model
    transform = AddGaussianNoise(mean=0.0, std=0.05) if hyperparameters['data_augmentation'] else None
    datasets_5d = {
        'train': Gaussian5DDataset(
            n_features=hyperparameters['n_features'], 
            distributions=hyperparameters['distributions'], 
            transform=transform
        ),
        'val': Gaussian5DDataset(
            n_features=hyperparameters['n_features'], 
            distributions=hyperparameters['distributions']
        )
    }

    test_ds = make_datasets_5d(hyperparameters['n_features_test'], hyperparameters['distributions'], for_test=True)
    model = MLP_5d()

    # Train model
    train_results = train(model, datasets_5d, hyperparameters)
    
    # Log results
    log_results(train_results, hyperparameters['output_dir'])
    
    # Test model
    test_model(train_results, test_ds, hyperparameters['output_dir'])


def train(model, datasets_5d, hyperparameters):

    optimizer = optim.Adam(
        model.parameters(), 
        lr=hyperparameters['learning_rate'], 
        weight_decay=hyperparameters['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=hyperparameters['epochs'],
        eta_min=hyperparameters['eta_min']
    )

    # Training loss criterion
    criterion_train = (
        make_weighted_loss(datasets_5d['train'], hyperparameters['alpha']) 
        if hyperparameters['use_weighted_loss_train'] 
        else nn.CrossEntropyLoss()
    )
    
    # Validation loss criteria: Cross-Entropy and custom loss
    criterion_val_ce = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    val_ce_loss_history = []  # To track Cross-Entropy loss during validation

    data_loaders = make_dataloaders(
        datasets_5d, 
        hyperparameters['use_train_sampler'], 
        hyperparameters['use_val_sampler'],
        hyperparameters['beta']
    )

    best_val_loss = float('inf')
    best_val_ce_loss = float('inf')  # To track the best CE loss
    best_model_state = None
    best_model_ce_state = None
    best_model_epoch = -1
    best_model_ce_epoch = -1

    for epoch in tqdm(range(hyperparameters['epochs'])):

        # Training step
        model.train()
        train_loss = 0.0
        for inputs, labels in data_loaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_train(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(data_loaders['train'])
        train_loss_history.append(train_loss)

        os.makedirs(hyperparameters['output_dir'], exist_ok=True)

        # Validation step
        model.eval()
        all_labels, all_predictions = [], []
        val_ce_loss = 0.0

        with torch.no_grad():
            for inputs, labels in data_loaders['val']:
                outputs = model(inputs)
                
                # Compute Cross-Entropy loss
                loss_ce = criterion_val_ce(outputs, labels)
                val_ce_loss += loss_ce.item()
                
                # Predictions for custom validation loss
                _, preds = torch.max(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        val_ce_loss /= len(data_loaders['val'])
        val_ce_loss_history.append(val_ce_loss)

        # Compute custom validation loss
        val_loss = validation_fnc(all_labels, all_predictions)
        val_loss_history.append(val_loss)

        # Save the best model based on custom validation loss
        if val_loss < best_val_loss:
            best_model_epoch = epoch
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_model_path = os.path.join(hyperparameters['output_dir'], 'best_model.pth')
            torch.save(best_model_state, best_model_path)

        # Save the best model based on CE loss
        if val_ce_loss < best_val_ce_loss:
            best_model_ce_epoch = epoch
            best_val_ce_loss = val_ce_loss
            best_model_ce_state = model.state_dict()
            best_model_ce_path = os.path.join(hyperparameters['output_dir'], 'best_model_ce.pth')
            torch.save(best_model_ce_state, best_model_ce_path)

        scheduler.step()
    
    

    # Save last model
    last_model_path = os.path.join(hyperparameters['output_dir'], 'last_model.pth')
    last_model_state = model.state_dict()
    torch.save(last_model_state, last_model_path)        
            
    results = {
        'best_val_loss': best_val_loss,
        'Cohens Kappa Score': 1 - best_val_loss,
        'best_model_path': best_model_path,
        'best_model_epoch': best_model_epoch,
        'best_val_ce_loss': best_val_ce_loss,
        'best_model_ce_path': best_model_ce_path,
        'best_model_ce_epoch': best_model_ce_epoch,
        'last_model_path': last_model_path
    }

    train_results = {
        'model': model,
        'best_model_path': best_model_path,
        'best_model_epoch': best_model_epoch,
        'best_val_loss': best_val_loss,
        'best_model_ce_path': best_model_ce_path,
        'best_model_ce_epoch': best_model_ce_epoch,
        'best_val_ce_loss': best_val_ce_loss,
        'last_model_path': last_model_path,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'val_ce_loss_history': val_ce_loss_history,
        'metadata': hyperparameters,
        'results': results
    }

    return train_results


def custom_validation_fnc(cm):
    # Convert the confusion matrix to a NumPy array
    cm = np.array(cm, dtype=float)

    # Square only the off-diagonal elements
    cm_off_diag = np.square(cm) * (1 - np.eye(cm.shape[0]))
    cm_diag = np.diag(np.diag(cm))
    cm_modified = cm_off_diag + cm_diag

    # Calculate the total number of elements in the squared matrix
    num_elements = np.sum(cm_modified)

    # Normalize the modified confusion matrix
    cm_normalized = cm_modified / num_elements
    print("Normalized Confusion Matrix:\n", cm_normalized)

    # Calculate the loss using the normalized matrix
    loss = 1 - np.trace(cm_normalized) + cm_normalized[1, 0] * cm_normalized[2, 0] \
           + cm_normalized[0, 1] * cm_normalized[2, 1] + cm_normalized[0, 2] * cm_normalized[1, 2]
    return loss

def validation_fnc(labels, predictions):
    """
    Calculate Cohen's Kappa score for validation.
    """
    return 1 - cohen_kappa_score(labels, predictions)


def test_model(train_results, test_ds, output_dir='./logs/'):
    os.makedirs(output_dir, exist_ok=True)
    model = train_results['model']
    
    # Test the best model (custom loss)
    best_model_path = train_results['best_model_path']
    state_dict = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    loader = make_dataloaders(test_ds, use_train_sampler=False, use_val_sampler=False, for_test=True)
    
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(preds.numpy())
    
    # Save confusion matrix and report for best model
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(3)))
    cm_path = os.path.join(output_dir, "confusion_matrix_best.jpg")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Best Model")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2"], digits=2)
    report_path = os.path.join(output_dir, "Classification_report_best.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Test the last model
    last_model_path = train_results['last_model_path']
    state_dict = torch.load(last_model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(preds.numpy())
    
    # Save confusion matrix and report for last model
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(3)))
    cm_path = os.path.join(output_dir, "confusion_matrix_last.jpg")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Last Model")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2"], digits=2)
    report_path = os.path.join(output_dir, "Classification_report_last.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Test the best model based on CE loss
    best_model_ce_path = train_results['best_model_ce_path']
    state_dict = torch.load(best_model_ce_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(preds.numpy())

    # Save confusion matrix and report for best CE model
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(3)))
    cm_path = os.path.join(output_dir, "confusion_matrix_best_ce.jpg")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Best CE Model")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2"], digits=2)
    report_path = os.path.join(output_dir, "Classification_report_best_ce.txt")
    with open(report_path, "w") as f:
        f.write(report)



def log_results(train_results, output_dir='./logs/'):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_results['train_loss_history'], label='Train Loss')
    plt.plot(train_results['val_loss_history'], label='Cohen-Kappa Score Validation Loss')
    plt.plot(train_results['val_ce_loss_history'], label='Cross-Entropy Validation Loss')
    
    # Dot for the Best Model (Cohen-Kappa Loss)
    plt.scatter(
        train_results['best_model_epoch'], 
        train_results['best_val_loss'], 
        color='red', 
        label='Best Model (Kappa)', 
        zorder=3
    )

    # Dot for the Best CE Model
    plt.scatter(
        train_results['best_model_ce_epoch'], 
        train_results['best_val_ce_loss'], 
        color='purple', 
        label='Best Model (CE)', 
        zorder=3
    )


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'train_results.png'))
    plt.close()

    combined_results = {
        'metadata': train_results['metadata'],
        'results': train_results['results']
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as metadata_file:
        json.dump(combined_results, metadata_file, indent=4)



def make_weighted_loss(ds, alpha=1.0):
    if isinstance(ds, Subset):
        dataset = ds.dataset
        indices = ds.indices
        labels = dataset.labels[indices]
    else:
        labels = ds.labels

    class_counts = torch.bincount(labels)
    class_weights = 1 / (class_counts.float())
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.pow(class_weights, alpha)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion


def make_random_sampler(ds, beta=1.0):
    if isinstance(ds, Subset):
        dataset = ds.dataset
        indices = ds.indices
        labels = dataset.labels[indices]
    else:
        labels = ds.labels

    class_counts = torch.bincount(labels)
    class_weights = 1 / (class_counts.float())
    class_weights = torch.pow(class_weights, beta)
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
    return sampler



def make_dataloaders(datasets_5d, use_train_sampler, use_val_sampler, beta = 1.0, for_test = False):
    data_loaders = {}

    if for_test:
        test_loader = DataLoader(
            datasets_5d,
            batch_size=4,
        )
        return test_loader

    train_loader = DataLoader(
        datasets_5d['train'],
        batch_size=4,
        shuffle=not use_train_sampler,
        sampler=make_random_sampler(datasets_5d['train'], beta) if use_train_sampler else None,
        drop_last=True
    )

    val_loader = DataLoader(
        datasets_5d['val'],
        batch_size=4,
        shuffle=not use_val_sampler,
        sampler=make_random_sampler(datasets_5d['val'], beta) if use_val_sampler else None,
        drop_last=True
    )

    data_loaders['train'] = train_loader
    data_loaders['val'] = val_loader

    return data_loaders


def make_datasets_5d(n_features, distributions, for_test = False):
    datasets = {}
    ds = Gaussian5DDataset(n_features, distributions)
    if for_test:
        return ds
    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.10, 
        shuffle=True, 
    )
    
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)
    
    datasets['train'] = train_ds
    datasets['val'] = val_ds
    
    return datasets


class MLP_5d(nn.Module):
    def __init__(self):
        super(MLP_5d, self).__init__()

        self.input_layer = nn.Linear(5, 12)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(12,3)
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x


class Gaussian5DDataset(Dataset):
    def __init__(self, n_features, distributions, transform=None):
        """
        Args:
            n_features: List of sample counts for each class.
            distributions: Tuple of means and standard deviations for each class.
            transform: Optional callable, applied to each sample.
        """
        self.data = []
        self.labels = []
        self.transform = transform  # Transformation function

        for label, (u, o) in enumerate(zip(*distributions)):
            samples = torch.normal(mean=u, std=o, size=(n_features[label], 5))
            self.data.append(samples)
            self.labels.append(torch.full((n_features[label],), label, dtype=torch.long))
        
        self.data = torch.cat(self.data)
        self.labels = torch.cat(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        if self.transform:
            sample = self.transform(sample)  # Apply transform if available
        return sample, label

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.normal(self.mean, self.std, size=tensor.size())


if __name__ == "__main__":
    main()
