import torch
from torch import nn  
import torch.nn.functional as F  

# Recurrent neural network with LSTM (many-to-one)
class LSTM_Model(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, num_lstm_layers, num_classes, device):
        super(LSTM_Model, self).__init__()
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(64*3*3, 128)
        
        self.lstm = nn.LSTM(128, lstm_hidden_size, num_lstm_layers, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        
        # Declare c0 and h0 hidden states for start lstm sequence
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device)
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device)
        
        batch_size, timesteps, H, W = x.size()
        x = x.view(batch_size*timesteps,1,H,W)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        x = x.view(batch_size*timesteps,-1)
        
        x = F.relu(self.fc1(x))
        
        x = x.view(batch_size,timesteps,-1)

        x, _ = self.lstm(x, (h0, c0)) 

        x = x[:, -1, :]
        x = x.reshape(x.shape[0], -1)

        x = self.fc2(x)
        return x

