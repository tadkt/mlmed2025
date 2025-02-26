import torch
import torch.nn as nn

class LSTM_classification(nn.Module):
    def __init__(self):
        super(LSTM_classification, self).__init__()
        self.lstm1 = nn.LSTM(input_size=187, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 5)
        
    def forward(self, x):
        x = x.unsqueeze(1) # for sequence_length 1
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1]
        x = self.fc(x)
        return x