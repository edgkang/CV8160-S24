import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, bias=True) -> None:
        super(MyRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, hidden = self.rnn(x, h0)
        out = self.fc(self.relu(out[:, -1, :]))

        return out

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, bias=True) -> None:
        super(MyLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, hidden = self.lstm(x, (h0, c0))
        out = self.fc(self.relu(out[:, -1, :]))

        return out
    

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, bias=True) -> None:
        super(MyGRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, hidden = self.lstm(x, h0)
        out = self.fc(self.relu(out[:, -1, :]))

        return out