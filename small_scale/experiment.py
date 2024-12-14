import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def main():

    # Hyperparameters
    n_features = [500, 500, 500]
    distributions = [
        [0, 3, -2],  # means
        [1, 1.5, 0.5]  # standard deviations
    ]
    learning_rate = 0.001
    weight_decay = 0.00005
    epochs = 40
    use_train_sampler = False
    use_val_sampler = False
    output_dir = './figs/'

    datasets_5d = make_datasets_5d(n_features, distributions)
    model = MLP_5d()
    train_results = train(model, datasets_5d, learning_rate, weight_decay, epochs, use_train_sampler, use_val_sampler, output_dir)
    log_results(train_results, output_dir)
    test_model(train_results, make_datasets_5d(n_features, distributions,for_test=True), output_dir)
    
def test_model(train_results, test_ds, output_dir='./logs/'):
    os.makedirs(output_dir, exist_ok=True)
    model = train_results['model']
    model.eval()
    loader = make_dataloaders(test_ds, use_train_sampler=False, use_val_sampler=False, for_test=True)

    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(preds.numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(3)))
    cm_path = os.path.join(output_dir, "confusion_matrix.jpg")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    report = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2"], digits=2)
    report_path = os.path.join(output_dir, f"Classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)


def log_results(train_results, output_dir='./logs/'):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_results['train_loss_history'], label='Train Loss')
    plt.plot(train_results['val_loss_history'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'train_results.png'))
    plt.close()

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as metadata_file:
        json.dump(train_results['metadata'], metadata_file, indent=4)

def train(model, datasets_5d, learning_rate, weight_decay, epochs, use_train_sampler, use_val_sampler, output_dir):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loss_history = []
    val_loss_history = []

    data_loaders = make_dataloaders(datasets_5d, use_train_sampler, use_val_sampler)

    for epoch in range(epochs):

        if epoch % 25 == 0:
            print(f"Training progress: {epoch} of {epochs}.")

        # train
        model.train()
        train_loss = 0.0
        for inputs, labels in data_loaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(data_loaders['train'])
        train_loss_history.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in data_loaders['val']:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(data_loaders['val'])
            val_loss_history.append(val_loss)
        
    metadata = {
        'learning_rate' : learning_rate,
        'weight_decay' : weight_decay,
        'epochs' : epochs,
        'use_train_sampler' : use_train_sampler,
        'use_val_sampler' : use_val_sampler,
        'final_val_loss' : val_loss,
        'output_dir' : output_dir
    }

    train_results = {
        'model' : model,
        'train_loss_history' : train_loss_history,
        'val_loss_history' : val_loss_history,
        'metadata' : metadata
    }

    return train_results


def make_random_sampler(ds):

    if isinstance(ds, Subset):
        dataset = ds.dataset
        indices = ds.indices
        labels = dataset.labels[indices]
    else:
        labels = ds.labels

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)

    return sampler


def make_dataloaders(datasets_5d, use_train_sampler, use_val_sampler, for_test = False):
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
        sampler=make_random_sampler(datasets_5d['train']) if use_train_sampler else None,
        drop_last=True
    )

    val_loader = DataLoader(
        datasets_5d['val'],
        batch_size=4,
        shuffle=not use_val_sampler,
        sampler=make_random_sampler(datasets_5d['val']) if use_val_sampler else None,
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
        test_size=0.33, 
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
    def __init__(self, n_features, distributions):
        self.data = []
        self.labels = []
        for label, (u, o) in enumerate(zip(*distributions)):
            samples = torch.normal(mean=u, std=o, size=(n_features[label], 5))
            self.data.append(samples)
            self.labels.append(torch.full((n_features[label],), label, dtype=torch.long))
        self.data = torch.cat(self.data)
        self.labels = torch.cat(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    main()
