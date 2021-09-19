from typing import Union, List
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from models.Conv1dLSTMAutoencoder import Conv1dLSTMAutoencoder
import Utils
import logging


class Model:
    def __init__(self, path: pathlib.Path, seq_len: int, n_features: int, h_conv_channel: Union[List[int], int],
                 kernel: Union[List[int], int], kernel_stride: Union[List[int], int], n_lstm: int, embedding_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.n_features = n_features
        self.h_conv_channel = h_conv_channel
        self.kernel = kernel
        self.kernel_stride = kernel_stride
        self.n_lstm = n_lstm
        self.embedding_dim = embedding_dim
        self.path = path / self.get_model_name()
        self.autoencoder = Conv1dLSTMAutoencoder(seq_len, n_features, h_conv_channel, kernel, kernel_stride,
                                                 embedding_dim, n_lstm)
        self.autoencoder.to(self.device)
        self.n_epochs = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_model_name(self):
        return "_".join([
            f"ch{Utils.to_sep_str(self.h_conv_channel, '-')}",
            f"k{Utils.to_sep_str(self.kernel, '-')}",
            f"ks{Utils.to_sep_str(self.kernel_stride, '-')}",
            f"lstm{self.n_lstm}",
            f"emb{self.embedding_dim}"
        ])

    def __str__(self):
        return str(self.autoencoder)

    def get_model_path(self):
        return self.path

    def save(self):
        self.path.mkdir(exist_ok=True, parents=True)
        with open(self.path / "metadata.txt", "w") as file:
            file.write(f"Convolution Channel: \t{Utils.to_sep_str(self.h_conv_channel, ', ')}\n".expandtabs(30))
            file.write(f"Kernel: \t{Utils.to_sep_str(self.kernel, ', ')}\n".expandtabs(30))
            file.write(f"Kernel Stride: \t{Utils.to_sep_str(self.kernel_stride, ', ')}\n".expandtabs(30))
            file.write(f"LSTM Layer: \t{self.n_lstm}\n".expandtabs(30))
            file.write(f"Embedding Dimension: \t{self.embedding_dim}\n".expandtabs(30))
            file.write(f"Epochs: \t{self.n_epochs}".expandtabs(30))
        self.logger.info(f"Saving model to {str(self.path / 'state_dict.pt')}")
        torch.save(self.autoencoder.state_dict(), self.path / "state_dict.pt")

    def train(self, dataset: Dataset, n_epochs: int, learning_rate: float, batch_size: int = 100, verbose: bool = True):
        self.logger.info("Starting training...")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        Utils.train_model(n_epochs, None, self.autoencoder, data_loader, loss_function, optimizer, self.device,
                          verbose=verbose, n_intra_epoch_log=3)
        self.n_epochs += n_epochs

    def predict(self, dataset: Dataset, indices: Union[List[int], None] = None, batch_size: int = 1000):
        if indices is not None:
            dataset = Subset(dataset, indices)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        output = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                output.append(self.autoencoder(sample_batched.to(self.device)).cpu())
        return torch.vstack(output).cpu()

    def reconstruction_error(self, dataset: Dataset, indices: Union[List[int], None] = None, batch_size: int = 1000):
        if indices is not None:
            dataset = Subset(dataset, indices)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.MSELoss(reduction="none")
        error = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                sample_batched = sample_batched.to(self.device)
                output_batched = self.autoencoder(sample_batched)
                error.append(torch.mean(loss(sample_batched, output_batched), dim=[1, 2]))
        return torch.hstack(error).cpu()

    def reconstruction_error_cobamas_max(self, dataset: Dataset, indices: Union[List[int], None] = None,
                                         batch_size: int = 1000):
        if indices is not None:
            dataset = Subset(dataset, indices)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.MSELoss(reduction="none")
        error = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                sample_batched = sample_batched.to(self.device)
                output_batched = self.autoencoder(sample_batched)
                # batch, seq_len, n_features * n_plants
                sample = loss(sample_batched, output_batched).reshape(-1, self.seq_len, 4, self.n_features // 4)
                sample = torch.max(torch.mean(sample, dim=[1, 2]), dim=1).values
                error.append(sample)
        return torch.hstack(error).cpu()

    def predict_batch(self, batch: torch.Tensor):
        with torch.no_grad():
            output_batched = self.autoencoder(batch.to(self.device))
        return output_batched.cpu()

    def reconstruction_error_batch(self, batch: torch.Tensor, output_batch: torch.Tensor):
        loss = nn.MSELoss(reduction="none")
        with torch.no_grad():
            error = torch.mean(loss(batch.to(self.device), output_batch.to(self.device)), dim=[1, 2])
        return error.cpu()
