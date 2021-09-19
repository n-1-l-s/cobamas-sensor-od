import torch
import torch.nn as nn


class LSTMEncoder2(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, hidden_dim=128):
        super(LSTMEncoder2, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = nn.LSTM(self.n_features, self.hidden_dim, batch_first=True)
        self.lstm_embedding = nn.LSTM(self.hidden_dim, self.embedding_dim, batch_first=True)

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features))
        x, (_, _) = self.lstm_hidden(x)
        _, (x, _) = self.lstm_embedding(x)
        return x.squeeze(0)


class LSTMDecoder2(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, hidden_dim=128):
        super(LSTMDecoder2, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_embedding = nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.lstm_hidden = nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.linear_output = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        x = x.unsqueeze(dim=1).repeat_interleave(self.seq_len, dim=1)
        x, (_, _) = self.lstm_embedding(x)
        x, (_, _) = self.lstm_hidden(x)
        # batch, seq_len, hidden_dim -> batch, seq_len, n_features
        x = self.linear_output(x)
        return x


class LSTMAutoencoder2(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, hidden_dim=128):
        super(LSTMAutoencoder2, self).__init__()
        self.encoder = LSTMEncoder2(seq_len, n_features, embedding_dim, hidden_dim)
        self.decoder = LSTMDecoder2(seq_len, n_features, embedding_dim, hidden_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
