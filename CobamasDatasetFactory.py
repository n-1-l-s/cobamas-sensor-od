from typing import Union, List
import pathlib
import numpy as np
from datasets.SensorData import SensorDataTimeOrig
import Utils
import logging


class CobamasDatasetFactory:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self, path):
        self.logger.info("Loading data...")
        with np.load(path) as npz:
            converters = npz["converters"]
            data = npz["data"]
            sensors = npz["sensors"]
            time = npz["time"]
        return data, time, sensors, converters

    def get_index_map(self, path, data, window_size, stride, plants, plant_names):
        self.logger.info("Preparing data...")
        try:
            self.logger.info(f"Looking for prepared data at {str(path / 'index.npz')}")
            with np.load(path / "index.npz") as index:
                index_map = index["arr_0"]
                self.logger.info(f"Data found")
        except FileNotFoundError:
            self.logger.info(f"No prepared data found. Creating data...")
            index_map = Utils.plant_index_map(data, window_size, stride, plants)
            self.logger.info(f"Saving prepared data at {str(path / 'index.npz')}")
            np.savez(path / "index.npz", index_map)
        if not (path / "metadata.txt").is_file():
            with open(path / "metadata.txt", "w") as file:
                file.write(f"Window Size: \t{window_size}\n".expandtabs(30))
                file.write(f"Stride: \t{stride}\n".expandtabs(30))
                file.write(f"Plant: \t{Utils.to_sep_str(plant_names, ', ')}\n".expandtabs(30))
        return index_map

    def get_dataset(self, data_path, window_size, stride, plants, scale="normalize", path="run"):
        data, time, sensors, converters = self.load_data(data_path)
        plant_names = list(map(lambda x: converters[x], plants))
        path = pathlib.Path(f"{path}/p{Utils.to_sep_str(plant_names)}_w{window_size}_s{stride}")
        self.logger.info(f"Creating dataset at {str(path)}")
        path.mkdir(exist_ok=True, parents=True)
        index_map = self.get_index_map(path, data, window_size, stride, plants, plant_names)
        if not isinstance(plants, list):
            plants = [plants]
        # data has dimension (time, sensor, plant)
        # transpose to get (time, plant, sensor)
        # data must be 2d, thus reshape to (time, plant*sensor)
        sensor_data = SensorDataTimeOrig(
            data=data[:, :, plants].transpose(0, 2, 1).reshape(data.shape[0], -1).astype(np.float32),
            indexer=index_map, window_size=window_size, stride=stride,
            transform=Utils.ToTensor(), scale=scale, time=time,
            sensor_names=sensors, converter_names=converters[plants])
        return sensor_data, window_size, len(plants) * len(sensors), path

    def __call__(self, data_path: str, save_path: str, window_size: int, stride: int, plants: Union[List[int], int],
                 scale: str = "normalize"):
        return self.get_dataset(data_path, window_size, stride, plants, scale, save_path)
