import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from tqdm import trange

from tdc.multi_pred import DDI
from tdc.utils import get_label_map

from tdc.benchmark_group import admet_group
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import numpy as np

import matplotlib.pyplot as plt

from torch.nn import Dropout

import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
import numpy as np

from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F

seed = 1
data = DDI(name = 'DrugBank')
split = data.get_split()
df = get_label_map(name = 'DrugBank', task = 'DDI', output_format='df')
df.head(25)
#print(df[df['Y'] == 1])
#get_label_map(name = 'DrugBank', task = 'DDI')

data.print_stats()

df['Y'] -= 1

import matplotlib.pyplot as plt

# Plot the class distribution
class_counts = df['Y'].value_counts().sort_index()
plt.figure(figsize=(12,6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.show()

import numpy as np
import pandas as pd

# Define the minimum number of samples you want for each class
desired_min_samples = 20000  # You can adjust this number based on your dataset

# Find classes that have fewer than the desired number of samples
classes_to_oversample = class_counts[class_counts < desired_min_samples].index

# Create an empty list to store augmented data
augmented_data = []

for cls in classes_to_oversample:
    # Get all samples of the minority class
    class_samples = df[df['Y'] == cls]
    num_samples = len(class_samples)
    num_copies = int(np.ceil(desired_min_samples / num_samples))

    # Duplicate the samples
    duplicated_samples = pd.concat([class_samples] * num_copies, ignore_index=True)

    # If we have more samples than desired_min_samples, truncate the excess
    duplicated_samples = duplicated_samples.sample(n=desired_min_samples, random_state=seed)

    augmented_data.append(duplicated_samples)

# Combine all augmented data with the original data excluding the oversampled classes
df_majority = df[~df['Y'].isin(classes_to_oversample)]
df_augmented = pd.concat([df_majority] + augmented_data, ignore_index=True)

# Shuffle the dataset
df_augmented = df_augmented.sample(frac=1, random_state=seed).reset_index(drop=True)

# Verify the new class distribution
new_class_counts = df_augmented['Y'].value_counts().sort_index()
print("New class distribution after oversampling:")
print(new_class_counts)

from sklearn.model_selection import train_test_split

# Split the augmented dataset
train_df, temp_df = train_test_split(
    df_augmented, test_size=0.2, random_state=seed, stratify=df_augmented['Y']
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=seed, stratify=temp_df['Y']
)

# Verify the distribution in each split
print("Training set class distribution:")
print(train_df['Y'].value_counts().sort_index())
print("\nValidation set class distribution:")
print(valid_df['Y'].value_counts().sort_index())
print("\nTest set class distribution:")
print(test_df['Y'].value_counts().sort_index())

from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

# Function to convert a SMILES string to a graph data object
def mol_to_graph_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid molecules

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),  # Number of attached hydrogens
        ]
        atom_features.append(features)
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])  # Undirected graph

        # Bond features
        bond_type = bond.GetBondType()
        bond_features = [
            bond_type == rdchem.BondType.SINGLE,
            bond_type == rdchem.BondType.DOUBLE,
            bond_type == rdchem.BondType.TRIPLE,
            bond_type == rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]
        edge_attr.extend([bond_features, bond_features])  # Add for both directions

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Function to merge two graphs into one
def merge_graphs(data1, data2):
    # Adjust edge indices for the second graph
    offset = data1.x.size(0)
    data2_edge_index = data2.edge_index + offset

    # Combine node features
    x = torch.cat([data1.x, data2.x], dim=0)

    # Combine edge indices
    edge_index = torch.cat([data1.edge_index, data2_edge_index], dim=1)

    # Combine edge attributes
    edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)

    # Create combined data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Featurization function for the dataset
def featurize_data(df):
    data_list = []
    for idx, row in df.iterrows():
        data1 = mol_to_graph_data(row['X1'])  # SMILES of Drug1
        data2 = mol_to_graph_data(row['X2'])  # SMILES of Drug2
        y = row['Y']

        if data1 is None or data2 is None:
            continue  # Skip invalid molecules

        # Merge the two drug graphs
        data = merge_graphs(data1, data2)
        data.y = torch.tensor([y], dtype=torch.long)  # For CrossEntropyLoss

        data_list.append(data)
    return data_list

# Featurize the datasets
train_data_list = featurize_data(train_df)
valid_data_list = featurize_data(valid_df)
test_data_list = featurize_data(test_df)

from torch_geometric.loader import DataLoader

batch_size = 1024

train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

print(f"number of samples in training: {len(train_loader)}")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class GCN(torch.nn.Module):
    def __init__(self, num_classes=86, dropout_rate=0.1):
        super(GCN, self).__init__()
        atom_feature_dim = 6
        edge_attr_dim = 6

        self.edge_mlp = Sequential(Linear(edge_attr_dim, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINEConv(nn=Linear(atom_feature_dim, 512), edge_dim=64)
        self.bn1 = torch.nn.BatchNorm1d(512)

        self.conv2 = GINEConv(nn=Linear(512, 1024), edge_dim=64)
        self.bn2 = torch.nn.BatchNorm1d(1024)

        self.conv3 = GINEConv(nn=Linear(1024, 2048), edge_dim=64)
        self.bn3 = torch.nn.BatchNorm1d(2048)

        self.fc1 = Linear(2048, 1024)
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, num_classes)  # Output layer for multiclass classification

        self.dropout = Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.edge_mlp(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(F.relu(x))
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(F.relu(x))
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(F.relu(x))
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # Outputs logits for each class

        return x  # Return logits directly
# Recalculate Class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Extract labels from the training data
train_labels = [data.y.item() for data in train_data_list]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recalculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("class weights recalculated")

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#criterion = FocalLoss(weight=class_weights.to(device))

model = GCN(num_classes=86)

model.to(device)
print("model on device")


optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

#optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CyclicLR(
#    optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=10, mode='triangular2'
#)

epochs = 50  # You can adjust the number of epochs as needed

# Lists to store losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    train_correct = 0
    total_train_samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        # Accumulate loss and correct predictions
        train_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        train_correct += (preds == batch.y.view(-1)).sum().item()
        total_train_samples += batch.num_graphs

    train_loss /= total_train_samples
    train_accuracy = train_correct / total_train_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    valid_loss = 0
    val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            valid_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            val_correct += (preds == batch.y.view(-1)).sum().item()
            total_val_samples += batch.num_graphs

    valid_loss /= total_val_samples
    val_accuracy = val_correct / total_val_samples
    val_losses.append(valid_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, '
          f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Step the scheduler
    scheduler.step(valid_loss)

# Save the trained model
torch.save(model.state_dict(), 'model_DrugBank.pth')



