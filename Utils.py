import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging


logger = logging.getLogger("Utils")


class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)


def train_model(n_epochs, model_shape,
                model, data_loader, loss_function, optimizer,
                device, verbose=True, n_intra_epoch_log=0, name=None, path=None):
    n_total_steps = len(data_loader)
    model.train()
    for i_epoch, epoch in enumerate(range(n_epochs)):
        epoch_loss = 0
        for i_batch, sample_batched in enumerate(data_loader):
            if model_shape is not None:
                batch_features = sample_batched.reshape(model_shape).to(device)
            else:
                batch_features = sample_batched.to(device)
            output = model(batch_features)
            loss = loss_function(output, batch_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if verbose and n_intra_epoch_log > 0 and (i_batch + 1) % (n_total_steps // n_intra_epoch_log) == 0:
                logger.info(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i_batch + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        if name is not None and path is not None:
            path_ = f"{path}{name}_epoch{i_epoch}.pt"
            torch.save(model.state_dict(), path_)
        if verbose:  # and (i_epoch + 1) % 200 == 0:
            logger.info(f"Epoch [{epoch + 1}/{n_epochs}] Loss: {epoch_loss / n_total_steps}")
    return model


def draw_compare(model, data, data_shape, model_shape, idx, idx_2, device, show_individual=True):
    model_input_1, model_output_1 = get_sample(model, data, data_shape, model_shape, idx, device)
    model_input_2, model_output_2 = get_sample(model, data, data_shape, model_shape, idx_2, device)
    if show_individual:
        draw(model_input_1, model_output_1)
        draw(model_input_2, model_output_2)
    draw_2(model_input_1, model_output_1, model_input_2, model_output_2)


def draw_single(model, data, data_shape, model_shape, idx, device):
    model_input, model_output = get_sample(model, data, data_shape, model_shape, idx, device)
    draw(model_input, model_output)


def draw_single_ext(model, data, data_shape, model_shape, idx, device, sensors):
    model_input, model_output = get_sample(model, data, data_shape, model_shape, idx, device)
    orig_sample, time = data.get_orig_item(idx)
    draw_ext(model_input, model_output, orig_sample, time, sensors, idx)


def draw_ext(model_input, model_output, orig_sample, time, sensors, w_idx):
    with torch.no_grad():
        fig, ax = plt.subplots(2, 4, figsize=(10,5), sharex=True)
        for i, ax_ in enumerate(fig.get_axes()):
            idx = i % len(sensors)
            if i < 4:
                ax_.set_title(sensors[idx])
                ax_.plot(orig_sample[:,idx], c="blue", label="signal")
            else:
                ax_.plot(model_input[:,idx], c="blue", label="signal")
                ax_.plot(model_output[:,idx], c="red", label="reconstruction")
                ax_.set_ylim([0,1])
            ax_.set_xlabel("time")
            #ax_.set_ylabel(sensors[idx])
        plt.suptitle("%i - %s" %(w_idx, np.datetime_as_string(time, unit="s")))
        plt.tight_layout()


def get_sample(model, data, data_shape, model_shape, idx, device):
    model_input = data[idx].reshape(model_shape).to(device)
    model_output = model(model_input)
    model_input = model_input.reshape(data_shape).cpu()
    model_output = model_output.reshape(data_shape).cpu()
    return model_input, model_output


def draw(model_input, model_output):
    with torch.no_grad():
        fig, ax = plt.subplots(2, 2)
        for i, ax_ in enumerate(fig.get_axes()):
            ax_.plot(model_input[:, i], c="blue")
            ax_.plot(model_output[:, i], c="red")
        return fig


def draw_input(model_input):
    with torch.no_grad():
        fig, ax = plt.subplots(2, 2)
        for i, ax_ in enumerate(fig.get_axes()):
            ax_.plot(model_input[:, i], c="blue")
        return fig


def draw_2(model_input_1, model_output_1, model_input_2, model_output_2):
    with torch.no_grad():
        fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
        for i, ax_ in enumerate(fig.get_axes()[:4]):
            ax_.plot(model_input_1[:, i], c="blue")
            ax_.plot(model_output_1[:, i], c="red")
        for i, ax_ in enumerate(fig.get_axes()[4:]):
            ax_.plot(model_input_2[:, i], c="blue")
            ax_.plot(model_output_2[:, i], c="red")
        return fig


def get_reconstruction_error(model, data, model_shape, device, batch_size=100):
    model.eval()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    loss = nn.MSELoss(reduction="none")
    # loss = nn.L1Loss(reduction="none")
    reconstruction_error = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if model_shape is not None:
                batch_features = sample_batched.reshape(model_shape).to(device)
            else:
                batch_features = sample_batched.to(device)
            output = model(batch_features)
            reconstruction_error.append(torch.mean(torch.flatten(loss(output, batch_features), 1), 1, True).cpu())
    return torch.vstack(reconstruction_error).squeeze()


def plant_index_map(data: np.ndarray, window_size: int, stride: int, plant):
    dataset_len = (data.shape[0] - window_size) // stride + 1
    index_map = np.full(dataset_len, True)
    for i in range(dataset_len):
        stride_idx = i * stride
        sample = data[stride_idx: stride_idx + window_size, :, plant]
        if np.isnan(sample.sum()):
            index_map[i] = False
    index_map = np.nonzero(index_map)[0]
    return index_map


def all_index_map(data, window_size, stride):
    return plant_index_map(data, window_size, stride, slice(0, data.shape[2]))


def to_sep_str(x, sep: str = "-"):
    if isinstance(x, list):
        return sep.join([str(val) for val in x])
    return str(x)


def get_batch(dataset, indices):
    return torch.stack([dataset[i] for i in indices])
