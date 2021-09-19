from typing import Union, List, Tuple
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as f
from models.SimpleLSTMAutoencoder import LSTMEncoder, LSTMDecoder


class Conv1dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    def output_shape(self, l_in):
        """output_shape of this conv1d-layer for a sequence of length l_in
        :returns (out_channels, L_out)"""
        return self.out_channels, int((l_in - self.kernel_size) / self.stride) + 1

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class Conv1dUnitStack(nn.Module):
    def __init__(self, in_channel: int, h_conv_channel: List[int], kernel_size: List[int], stride: List[int]):
        super(Conv1dUnitStack, self).__init__()
        self.in_channel = in_channel
        h_conv_channel = [in_channel] + h_conv_channel
        self.conv_layers = [Conv1dUnit(h_conv_channel[i], h_conv_channel[i + 1], kernel_size[i], stride[i]) for i in
                            range(len(h_conv_channel) - 1)]
        self.model = nn.Sequential(*self.conv_layers)

    def output_shapes(self, l_in):
        """list of output_shapes for each layer of the stack, including layer 0 (i.e. the input of the first layer),
        for a sequence of length l_in
        :returns [(out_channels, L_out), ...]"""
        shapes = [(self.in_channel, l_in)]
        for conv1d_unit in self.conv_layers:
            shapes.append(conv1d_unit.output_shape(shapes[-1][1]))
        return shapes

    def output_shape(self, l_in):
        """output_shape of the whole conv1d stack for a sequence of length l_in
        :returns (L_out, out_channels)"""
        out_channel, l_out = self.in_channel, l_in
        for conv1d_unit in self.conv_layers:
            out_channel, l_out = conv1d_unit.output_shape(l_out)
        return l_out, out_channel

    def forward(self, x):
        # (Batch, L_in, out_channels) -> swap to (Batch, out_channels, L_in)
        x = x.transpose(1, 2)
        x = self.model(x)
        # swap back to (Batch, L_out, out_channels)
        x = x.transpose(1, 2)
        return x


class DeConv1dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConv1dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_transposed1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x, output_size):
        x = self.conv_transposed1d(x, output_size=output_size)
        return x


class DeConv1dUnitStack(nn.Module):
    def __init__(self, in_channel: int, output_size: List[Tuple[int, int]], kernel_size: List[int], stride: List[int]):
        super(DeConv1dUnitStack, self).__init__()
        self.output_size = output_size
        h_conv_channel = [in_channel] + [channel for channel, _ in self.output_size]
        self.conv_layers = nn.ModuleList(
            [nn.ConvTranspose1d(h_conv_channel[i], h_conv_channel[i + 1], kernel_size[i], stride[i]) for i in
             range(len(h_conv_channel) - 1)])

    def forward(self, x):
        # (Batch, L_in, out_channels) -> swap to (Batch, out_channels, L_in)
        x = x.transpose(1, 2)
        # conv_transposed1d cannot recreate the original input shape if stride > 1
        # thus, the original input shape has to be provided in the forward-function as output_size
        # x.shape[0] adds the batch size, as it its not known beforehand
        for conv_layer, (out_channel, l_out) in zip(self.conv_layers, self.output_size):
            x = conv_layer(x, output_size=(x.shape[0], out_channel, l_out))
        # swap back to (Batch, L_out, out_channels)
        x = x.transpose(1, 2)
        return x


class Conv1dLstmEncoder(nn.Module):
    def __init__(self, in_channel: int, h_conv_channel: List[int], kernel_size: List[int], stride: List[int],
                 embedding_dim: int, n_lstm_layer: int):
        super(Conv1dLstmEncoder, self).__init__()
        self.conv_unit_stack = Conv1dUnitStack(in_channel, h_conv_channel, kernel_size, stride)
        self.lstm_encoder = LSTMEncoder(h_conv_channel[-1], embedding_dim, n_lstm_layer)

    def forward(self, x):
        x = self.conv_unit_stack(x)
        x = self.lstm_encoder(x)
        return x


class Conv1dLstmDecoder(nn.Module):
    def __init__(self, in_channel: int, output_size: List[Tuple[int, int]], kernel_size: List[int], stride: List[int],
                 embedding_dim: int, n_lstm_layer: int):
        super(Conv1dLstmDecoder, self).__init__()
        self.lstm_decoder = LSTMDecoder(output_size[0][0], embedding_dim, output_size[0][1], n_lstm_layer)
        self.de_conv_unit_stack = DeConv1dUnitStack(in_channel, output_size[1:], kernel_size, stride)

    def forward(self, x):
        x = self.lstm_decoder(x)
        x = self.de_conv_unit_stack(x)
        return x


def _listify(x: Union[int, List[int]]):
    return [x] if isinstance(x, int) else x


def _extend_to(x: List, n: int):
    if n > 1 and len(x) == 1:
        x = x + (x * (n - 1))
    return x


def _to_aligned_lists(*args: Union[int, List[int]]):
    """Extends all arguments to lists of the same length"""
    args = list(map(_listify, args))
    n = max(map(len, args))
    if n == 1:
        return args
    args = list(map(partial(_extend_to, n=n), args))
    min_len = min(map(len, args))
    if min_len != n:
        raise ValueError("Cannot align arguments.\nArguments must be int, list of length 1 or list of equal length.")
    return args


class Conv1dLSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, in_channel: int, h_conv_channel: Union[int, List[int]],
                 kernel_size: Union[int, List[int]], stride: Union[int, List[int]],
                 embedding_dim: int, n_lstm_layer: int = 1):
        super(Conv1dLSTMAutoencoder, self).__init__()
        h_conv_channel, kernel_size, stride = _to_aligned_lists(h_conv_channel, kernel_size, stride)
        self.encoder = Conv1dLstmEncoder(in_channel, h_conv_channel, kernel_size, stride, embedding_dim, n_lstm_layer)
        output_size = self.encoder.conv_unit_stack.output_shapes(seq_len)
        self.decoder = Conv1dLstmDecoder(h_conv_channel[-1], output_size[::-1], kernel_size[::-1], stride[::-1],
                                         embedding_dim, n_lstm_layer)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
