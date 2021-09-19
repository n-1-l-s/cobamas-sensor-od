from CobamasDatasetFactory import CobamasDatasetFactory
from ModelFactory import ModelFactory
from CobamasVisualizer import CobamasVisualizer
from Model import Model
from Utils import get_batch
import logging
import matplotlib.pyplot as plt
import torch

logging.basicConfig(level=logging.INFO)

# dataset param
#DATA_PATH = "data/sensor_export_full.npz"
DATA_PATH = "/home/nils/Documents/ude/cobamas/data/sensor_export/sensor_export_full.npz"
DATASET_PATH = "run"
WINDOWS_SIZE = 180
STRIDE = 60
PLANTS = [3]
# model param
H_CONV_CHANNEL = 8
KERNEL = 7
KERNEL_STRIDE = 2
N_LSTM = 1
EMBEDDING_DIM = 32
# train param
N_EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
# eval param
INDICES = None

cdf = CobamasDatasetFactory()
dataset, seq_len, n_features, path = cdf(DATA_PATH, DATASET_PATH, WINDOWS_SIZE, STRIDE, PLANTS)
mf = ModelFactory(seq_len, n_features, path)
#model = mf.load_model(H_CONV_CHANNEL, KERNEL, KERNEL_STRIDE, N_LSTM, EMBEDDING_DIM)
model = mf.get_model(H_CONV_CHANNEL, KERNEL, KERNEL_STRIDE, N_LSTM, EMBEDDING_DIM)
model.train(dataset, N_EPOCHS, LEARNING_RATE, BATCH_SIZE, verbose=True)
model.save()

# output = model.predict(dataset, indices=INDICES)
error = model.reconstruction_error(dataset, indices=INDICES)

sort_idx = torch.argsort(error)
for i, idx in enumerate(sort_idx[:10]):
    fig = CobamasVisualizer.plot_multi_plant_sample(dataset, model, idx, save_path=model.get_model_path() / f"max_best_{i}.png")
    plt.close()
for i, idx in enumerate(sort_idx[-10:]):
    fig = CobamasVisualizer.plot_multi_plant_sample(dataset, model, idx, save_path=model.get_model_path() / f"max_worst_{i}.png")
    plt.close()
CobamasVisualizer.plot_error_distribution(error, save_path=model.get_model_path() / f"error_distribution_boxplot.png")
plt.show()
