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
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(num_features, 8)
        self.fc2 = nn.Linear(64, 8)
#         self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_fingers)
        
        self.relu = nn.ReLU()
        
        nn.init.xavier_normal_(self.fc1.weight)
#         nn.init.xavier_normal_(self.fc2.weight)
#         nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.15)
        
    def forward(self, x):
#         x = self.bn(x)
        x = self.dropout1(x)
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
        
#         x= self.dropout2(x)
        
#         x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        
        x = self.dropout2(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.relu(self.fc3(x))
#         x = self.dropout2(x)
        output = self.fc4(x)
        
        return output
    
class FingerFeatureDataset(Dataset):
    def __init__(self, R, dg, window=2000):
        self.R = np.float32(R)
#         self.R = (self.R - ecog_1_train.mean(axis=0)) / ecog_1_train.std(axis=0)
#         self.ecog = self.ecog.reshape(self.ecog.shape[0], 1, -1)
        self.dg = np.float32(dg)
        
    

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        
        return self.R[idx], self.dg[idx]