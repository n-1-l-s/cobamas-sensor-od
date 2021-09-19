import pathlib
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import Utils
import logging
from CobamasDatasetFactory import CobamasDatasetFactory
from ModelFactory import ModelFactory
from CobamasVisualizer import CobamasVisualizer
from Model import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        return idx, self.dataset[idx]


class CobamasSynthesizer():
    def __init__(self):
        pass

    def find_greater(self, dataset, feature, threshold, max_n=100, batch_size=100):
        idx_dataset = IndexDataset(dataset)
        data_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False)
        output_idx = []
        output_value = []
        count = 0
        with torch.no_grad():
            for i_batch, (idx_batched, sample_batched) in enumerate(data_loader):
                sample_batched = torch.sum(sample_batched[:, :, feature], dim=1)
                mask = sample_batched >= threshold
                if torch.any(mask):
                    output_idx.append(idx_batched[mask])
                    output_value.append(sample_batched[mask])
                    count += output_idx[-1].shape[0]
                    if count > max_n:
                        break
        return torch.hstack(output_idx), torch.hstack(output_value)

    def set_feature(self, dataset, indices, feature, value, start=0):
        batch = Utils.get_batch(dataset, indices)
        return self.set_feature_batch(batch, feature, value, start)

    def set_feature_batch(self, batch, feature, value, start=0):
        batch[:, start:, feature] = value
        return batch

    def arg_find_closest_multiple(self, values, to):
        output = []
        for val in to:
            output.append(self.arg_find_closest(values, val))
        return torch.stack(output)

    def arg_find_closest(self, values, to):
        return torch.argmin(torch.abs(values - to))

    def swap_features(self, dataset, indices, feature_order):
        batch = Utils.get_batch(dataset, indices)
        return self.swap_features_batch(batch, feature_order)

    def swap_features_batch(self, batch, feature_order):
        return batch[:, :, feature_order]

    def drehzahl_0(self, dataset, model, n_windows, save_name=None):
        idx, value = self.find_greater(dataset, 2, 100, max_n=10000)
        closest_idx = self.arg_find_closest_multiple(value,
                                                     torch.linspace(torch.min(value).item(), torch.max(value).item(),
                                                                    n_windows))
        window_idx = idx[closest_idx]
        batch = Utils.get_batch(dataset, window_idx)
        transformed_windows = self.set_feature(dataset, window_idx, 0, 0)
        err = model.reconstruction_error(dataset, None)
        err_batch = model.reconstruction_error_batch(batch, model.predict_batch(batch))
        err_transformed = model.reconstruction_error_batch(transformed_windows,
                                                           model.predict_batch(transformed_windows))
        if save_name is not None:
            save_name = model.get_model_path() / save_name
        return CobamasVisualizer.plot_transformed_error_distribution(err, err_batch, err_transformed,
                                                                     title=f"Shifted Error\n{dataset.sensor_names[0]} = 0",
                                                                     save_path=save_name)

    def swap(self, dataset, model, n_windows, feature_order, save_name=None):
        error = model.reconstruction_error(dataset, None)
        window_idx = self.arg_find_closest_multiple(error,
                                                    torch.linspace(torch.min(error).item(), torch.max(error).item(),
                                                                   n_windows))
        batch = Utils.get_batch(dataset, window_idx)
        transformed_windows = cs.swap_features(dataset, window_idx, feature_order)
        err_batch = model.reconstruction_error_batch(batch, model.predict_batch(batch))
        err_transformed = model.reconstruction_error_batch(transformed_windows,
                                                           model.predict_batch(transformed_windows))
        if save_name is not None:
            save_name = model.get_model_path() / save_name
        return CobamasVisualizer.plot_transformed_error_distribution(error, err_batch, err_transformed,
                                                                     title=f"Shifted Error\nSwapped {Utils.to_sep_str(feature_order, '-')}",
                                                                     save_path=save_name)

    def kurzschluss(self, dataset, model, n_windows, start=0, save_name=None):
        idx, value = cs.find_greater(dataset, 1, 175, max_n=10000)
        closest_idx = cs.arg_find_closest_multiple(value,
                                                   torch.linspace(torch.min(value).item(), torch.max(value).item(),
                                                                  n_windows))
        window_idx = idx[closest_idx]
        batch = Utils.get_batch(dataset, window_idx)
        transformed_windows = cs.set_feature(dataset, window_idx, [1, 2, 3], 0, start=start)
        err = model.reconstruction_error(dataset, None)
        err_batch = model.reconstruction_error_batch(batch, model.predict_batch(batch))
        err_transformed = model.reconstruction_error_batch(transformed_windows,
                                                           model.predict_batch(transformed_windows))
        if save_name is not None:
            save_name = model.get_model_path() / save_name
        return CobamasVisualizer.plot_transformed_error_distribution(err, err_batch, err_transformed,
                                                                     title=f"Short Circuit {start}\n{Utils.to_sep_str([1, 2, 3], '-')} = 0",
                                                                     save_path=save_name)

    def multi_plant(self, dataset, model, n_windows, threshold, batch_size=100, save_name=None):
        error = model.reconstruction_error(dataset, None)
        window_idx = self.arg_find_closest_multiple(error,
                                                    torch.linspace(torch.min(error).item(), torch.max(error).item(),
                                                                   n_windows))
        batch = Utils.get_batch(dataset, window_idx)
        batch_transformed = Utils.get_batch(dataset, window_idx).reshape(n_windows, dataset.window_size, 4, -1)
        for i, idx in enumerate(window_idx):
            diff_idx, diff = [], []
            last_plant_values = dataset[idx].reshape(dataset.window_size, 4, -1)[:, :, -1]
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for i_batch, sample_batched in enumerate(data_loader):
                sample_batched = sample_batched.reshape(-1, dataset.window_size, 4, len(dataset.converter_names))[:, :, :, -1]
                batch_diff = torch.mean(torch.abs(sample_batched - last_plant_values.unsqueeze(0)), dim=[1, 2])
                idx_batch_closest = self.arg_find_closest(batch_diff, threshold).item()
                diff_idx.append(i_batch * batch_size + idx_batch_closest)
                diff.append(batch_diff[idx_batch_closest].item())
            swap_idx = diff_idx[self.arg_find_closest(torch.tensor(diff), threshold).item()]
            batch_transformed[i, :, :, -1] = dataset[swap_idx].reshape(dataset.window_size, 4, -1)[:, :, -1]
        batch_transformed = batch_transformed.reshape(n_windows, dataset.window_size, -1)
        err_batch = model.reconstruction_error_batch(batch, model.predict_batch(batch))
        err_transformed = model.reconstruction_error_batch(batch_transformed,
                                                           model.predict_batch(batch_transformed))
        if save_name is not None:
            save_name = model.get_model_path() / save_name
        return CobamasVisualizer.plot_transformed_error_distribution(error, err_batch, err_transformed,
                                                                     title=f"Swap last plant\ndiff {threshold}",
                                                                     save_path=save_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    WINDOWS_SIZE, STRIDE, PLANTS = 180, 60, [3]
    cdf = CobamasDatasetFactory()
    dataset, seq_len, n_features, path = cdf("data/sensor_export_full.npz", "run", WINDOWS_SIZE, STRIDE, PLANTS)
    mf = ModelFactory(seq_len, n_features, path)
    model = mf.load_model(8, 7, 2, 1, 32)
    cs = CobamasSynthesizer()
    #cs.drehzahl_0(dataset, model, 5, "drehzahl_0.pdf")
    order = [2, 1, 0, 3]
    cs.swap(dataset, model, 5, order, save_name=f"err_dist_swap_{Utils.to_sep_str(order, '')}.pdf")
    # start = 170
    # cs.kurzschluss(dataset, model, 5, start, save_name=f"kurzschluss_{start}.pdf")
    #thesholds = [0.1, 0.2, 0.3]
    #for theshold in thesholds:
    #    cs.multi_plant(dataset, model, 10, theshold, save_name=f"swap_last_{theshold}_{10}.pdf")
    plt.show()
