from typing import Union, List
import pathlib
import torch
from Model import Model
import logging


class ModelFactory:
    def __init__(self, seq_len: int, n_features: int, path: Union[str, pathlib.Path]):
        """
        dataset is a pytorch dataset containing all samples
        seq_len, n_features defines the shape of each sample in dataset,
        path is the baseline path used to save all models using this dataset
        """
        self.seq_len = seq_len
        self.n_features = n_features
        if isinstance(path, str):
            self.path = pathlib.Path(path)
        else:
            self.path = path
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_model(self, h_conv_channel: Union[List[int], int], kernel: Union[List[int], int],
                  kernel_stride: Union[List[int], int], n_lstm: int, embedding_dim: int):
        return Model(self.path, self.seq_len, self.n_features, h_conv_channel, kernel, kernel_stride, n_lstm,
                     embedding_dim)

    def load_model(self, h_conv_channel: Union[List[int], int], kernel: Union[List[int], int],
                   kernel_stride: Union[List[int], int], n_lstm: int, embedding_dim: int):
        model = self.get_model(h_conv_channel, kernel, kernel_stride, n_lstm, embedding_dim)
        model_path = model.get_model_path()
        self.logger.info(f"Loading model from {str(model_path / 'state_dict.pt')}")
        model.autoencoder.load_state_dict(torch.load(model_path / "state_dict.pt"))
        return model
