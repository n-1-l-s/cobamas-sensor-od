from typing import Union
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SensorData(Dataset):
    def __init__(self, data: np.ndarray, indexer: np.ndarray, 
                 sensor_names: np.ndarray, converter_names: np.ndarray, window_size: int = 60, stride: int = 1,
                 transform=None, scale: Union[str, None] = "normalize"):
        super(SensorData, self).__init__()
        self.data = data
        self.sensor_names = sensor_names
        self.converter_names = converter_names
        self.indexer = indexer
        self.scaler = None
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        if scale is not None:
            if scale == "standardize":
                self.scaler = StandardScaler()
            elif scale == "normalize":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling option: {scale}\nscale must be 'standardize', 'normalize' or None")
            self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return self.indexer.shape[0]

    def neg_idx(self, idx):
        if idx < 0:
            return len(self) + idx
        return idx

    def __getitem__(self, idx):
        idx = self.neg_idx(idx)
        stride_idx = self.indexer[idx] * self.stride
        sample = self.data[stride_idx: stride_idx + self.window_size]
        if self.transform:
            sample = self.transform(sample)
        return sample


class SensorDataTimeOrig(SensorData):
    def __init__(self, data: np.ndarray, time: np.ndarray, 
                 indexer: np.ndarray, sensor_names: np.ndarray, 
                 converter_names: np.ndarray, window_size: int = 60, 
                 stride: int = 1,
                 transform=None, scale: Union[str, None] = "normalize"):
        super(SensorDataTimeOrig, self).__init__(data, indexer, sensor_names, converter_names, window_size, stride, transform, scale)
        self.time = time
        self.orig_data = data

    def get_orig_item(self, idx):
        idx = self.neg_idx(idx)
        stride_idx = self.indexer[idx] * self.stride
        sample = self.orig_data[stride_idx: stride_idx + self.window_size]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.time[stride_idx]
