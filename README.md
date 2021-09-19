# CoBaMas Sensor Outlier Detection

### <p align=center>`CobamasSensorOD`</p>

CobamasSensorOD is a framework used to create, train and visualize an autoencoder on sequential multivariate data.
It uses PyTorch as a baseline framework for the model and provides a predefined model architecture and training loop.
The framework was created to analyse sensor data from wind turbines to find anomalous behaviour.

## Getting Started

### 1. Data

The predefined training loop requires a `torch.utils.data Dataset`. For your own data you need to implement your own dataset.
The only requirement is that the output sample is a 2d `tensor` of shape `(time_step, feature)`. 

### 2. Model

![Alt text](autoencoder_conv1d_lstm.png?raw=true "Model-Architecture")

The Encoder consists of two parts. The first part is a 1d-convolution-layer, where each feature is considered as one channel
and the second part is a lstm-layer used to create the embedding. The convolution-layer will shorten the input-sequence and
generate new features as additional channels. The input in the lstm-layer is the output of the convolution-layer.

The model can be used as Model-Object with predefined methods for loading, saving, training and evaluation.
```python
from Model import Model
from ModelFactory import ModelFactory

mf = ModelFactory(seq_len=300, n_features=4, path="run")
model = mf.get_model(
    h_conv_channel=[8, 12],
    kernel=7, 
    kernel_stride=[3,1],
    embedding_dim=32, 
    n_lstm=2
)
```
Alternatively the model can be used as a PyTorch module directly:
```python
from models.Conv1dLSTMAutoencoder import Conv1dLSTMAutoencoder

model = Conv1dLSTMAutoencoder(
    seq_len=300, 
    in_channel=4,
    h_conv_channel=[8, 12],
    kernel_size=7, 
    stride=[3,1],
    embedding_dim=32, 
    n_lstm_layer=2
)
```
### ModelFactory/Model Parameters
**ModelFactory**
- `seq_len` _(int)_: Number of time_steps per sample in the dataset. Equal to `seq_len` of the PyTorch-Module.
- `n_features` _(int)_: Number of features per time_step per sample in the dataset. Equal to `in_channel` of the PyTorch-Module.
Defines number of input channels for the first conv1d-layer.
- `path` _(str)_: Root directory for every file-based function such as loading/saving.

**Model**
- `h_conv_channel` _(list or int)_: Defines the number of output channels for each conv1d-layer. 
If list of length n, n-1 hidden conv1d-layers will be added to the model.
- `kernel` _(list or int)_: Defines the kernel size for each conv1d-layer.
- `kernel_stride` _(list or int)_: Defines the kernel stride for each conv1d-layer.
- `n_lstm` _(int)_: Number of lstm-layers. Must be >= 1.
- `embedding_dim` _(int)_: Number of dimensions of the embedding and thus the number of dimensions of the hidden state of the lstm-layer.

### Training
Using the predefined `Model`-class training can be achieved by calling the `train`-Method with a `torch.utils.data Dataset`.
```python
model.train(dataset=dataset, n_epochs=10, learning_rate=0.001, batch_size=100, verbose=True)
```
### Saving and Loading Models
The model can be saved via the `save`-method of the `Model`-class and loaded via the `load`-method of the `ModelFactory`-class.
The model will be saved in a folder based on the model-parameters and in the directory provided in the 
`path`-parameter of the `ModelFactory`-class.
```python
mf = ModelFactory(seq_len=300, n_features=4, path="run")
model = mf.get_model(h_conv_channel=8, kernel=7, kernel_stride=3, embedding_dim=32, n_lstm=1)
model.save()
model = mf.load_model(h_conv_channel=8, kernel=7, kernel_stride=3, embedding_dim=32, n_lstm=1)
```
##