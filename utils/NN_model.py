import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device("cpu")

class FingerRegressor(nn.Module):
    def __init__(self, num_features, num_fingers) -> None:
        super(FingerRegressor, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_fingers)
        self.relu = nn.ReLU()
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.15)
        
    def forward(self, x):
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)

        
        return output
    
class FingerFeatureDataset(Dataset):
    def __init__(self, R, dg, window=2000):
        self.R = np.float32(R)
        self.dg = np.float32(dg)
        
    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return self.R[idx], self.dg[idx]