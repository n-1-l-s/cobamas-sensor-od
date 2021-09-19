from typing import Union, List, Tuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as f


class LSTMHiddenUnit(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(LSTMHiddenUnit, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=True)
        self.activation = activation

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LSTMOutputUnit(LSTMHiddenUnit):
    def __init__(self, input_size, hidden_size, activation=None):
        super(LSTMOutputUnit, self).__init__(input_size, hidden_size, activation)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LSTMUnitStack(nn.Module):
    def __init__(self, input_size: int, h_lstm_size: List[int], h_activation=None):
        super(LSTMUnitStack, self).__init__()
        lstm_size = [input_size] + h_lstm_size
        layers = [LSTMHiddenUnit(lstm_size[i], lstm_size[i + 1], h_activation) for i in range(len(lstm_size) - 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, h_lstm_size: List[int], embedding: int, h_activation=None, out_activation=None):
        super(LSTMEncoder, self).__init__()
        if len(h_lstm_size) > 0:
            self.lstm_unit = nn.Sequential(LSTMUnitStack(input_size, h_lstm_size, h_activation),
                                           LSTMOutputUnit(h_lstm_size[-1], embedding, out_activation))
        else:
            self.lstm_unit = LSTMOutputUnit(input_size, embedding, out_activation)

    def forward(self, x):
        x = self.lstm_unit(x)
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, input_size: int, h_lstm_size: List[int], output_size: int, h_activation=None):
        super(LSTMDecoder, self).__init__()
        if len(h_lstm_size) > 0:
            self.lstm_unit = LSTMUnitStack(input_size, h_lstm_size[:-1] + h_lstm_size[-1:] * 2, h_activation)
            self.linear = nn.Linear(h_lstm_size[-1], output_size)
        else:
            self.lstm_unit = LSTMHiddenUnit(input_size, input_size, h_activation)
            self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.lstm_unit(x)
        x = self.linear(x)
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, h_lstm_size: List[int], embedding: int, h_activation=None, out_activation=None):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, h_lstm_size, embedding, h_activation, out_activation)
        self.decoder = LSTMDecoder(embedding, h_lstm_size[::-1], input_size, h_activation)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    foo = LSTMAutoencoder(32, [24, 22, 20], 12, h_activation=nn.ReLU)
    print(foo)
