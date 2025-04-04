import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- Define the Binding Site Siamese Network ---- #
class BindingSiteNet(nn.Module):
    def __init__(self, input_dim=5, output_dim=80):
        super(BindingSiteNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 80, kernel_size=3, padding=1)  # Convolutional Layer
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool2d((16, 5))  # Mimic soft grid mapping
        self.fc = nn.Linear(80 * 16 * 5, output_dim)  # Fully Connected Layer
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# ---- Contrastive Loss Function ---- #
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, d1, d2, label):
        distance = torch.nn.functional.pairwise_distance(d1, d2)
        loss = (label * distance**2) + ((1 - label) * torch.clamp(self.margin - distance, min=0.0) ** 2)
        return loss.mean()

# ---- Custom Dataset Class ---- #
class BindingSiteDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Shape: (num_samples, 2, input_dim, height, width)
        self.labels = labels  # Shape: (num_samples,)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        site1, site2 = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(site1, dtype=torch.float32), torch.tensor(site2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ---- Training Loop ---- #
def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for site1, site2, label in train_loader:
            site1, site2, label = site1.cuda(), site2.cuda(), label.cuda()
            
            optimizer.zero_grad()
            d1 = model(site1)
            d2 = model(site2)
            loss = criterion(d1, d2, label)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    
    return model

# ---- Example Usage ---- #
if __name__ == "__main__":
    # Generate random sample data (replace with real data)
    num_samples = 1000
    input_dim = 5
    height, width = 16, 5  # Soft grid mapping
    
    data = np.random.rand(num_samples, 2, input_dim, height, width)  # Pairs of binding sites
    labels = np.random.randint(0, 2, size=(num_samples,))  # Similar (1) or Dissimilar (0)
    
    dataset = BindingSiteDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model Initialization
    model = BindingSiteNet(input_dim=input_dim).cuda()
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Model
    trained_model = train_model(train_loader, model, criterion, optimizer, num_epochs=10)

