import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, self.embedding_dim)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, self.input_dim)
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(self.input_dim, self.embedding_dim)
        self.decoder = Decoder(self.input_dim, self.embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
