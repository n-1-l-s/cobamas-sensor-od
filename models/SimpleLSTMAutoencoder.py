import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_lstm_layer):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(n_features, embedding_dim, batch_first=True, num_layers=n_lstm_layer)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class LSTMDecoder(nn.Module):
    def __init__(self, n_features, embedding_dim, seq_len, n_lstm_layer):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, num_layers=n_lstm_layer)
        self.linear_output = nn.Linear(embedding_dim, n_features)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(dim=1).repeat_interleave(self.seq_len, dim=1)
        x, (_, _) = self.lstm(x)
        x = self.linear_output(x)
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim, seq_len, n_lstm_layer=1):
        super(LSTMAutoencoder, self).__init__()
        if n_lstm_layer < 1:
            raise ValueError("n_lstm_layer must be greater than 0")
        self.encoder = LSTMEncoder(n_features, embedding_dim, n_lstm_layer)
        self.decoder = LSTMDecoder(n_features, embedding_dim, seq_len, n_lstm_layer)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
