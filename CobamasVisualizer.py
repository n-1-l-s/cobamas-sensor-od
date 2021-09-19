import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Utils import get_batch
import numpy as np


class CobamasVisualizer:
    @classmethod
    def plot_multi_plant_sample(cls, dataset, model, idx, v_scale=2, h_scale=3, save_path=None):
        converters = dataset.converter_names
        sensors = dataset.sensor_names

        sample, time = dataset.get_orig_item(idx)
        batch = get_batch(dataset, [idx])
        recon = model.predict_batch(batch)
        recon = recon[0]
        sample_scaled = batch[0]

        fig, ax = plt.subplots(len(converters) + 1, len(sensors),
                               figsize=(h_scale * len(sensors), v_scale * (len(converters) + 1)), sharex=True)
        fig.patch.set_facecolor('white')
        axes = fig.get_axes()

        # first line: original values
        linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (5, 5))]
        for s_idx, sensor in enumerate(sensors):
            ax = axes[s_idx]
            ax.set_title(sensor)
            for c_idx, c in enumerate(converters):
                ax.plot(sample[:, c_idx * len(sensors) + s_idx], label=c, linestyle=linestyles[c_idx])
            if s_idx == 1:
                ax.legend()

        # reconstructions
        for c_idx, converter in enumerate(converters):
            for s_idx, sensor in enumerate(sensors):
                ax = axes[s_idx + (c_idx + 1) * len(sensors)]
                _idx = s_idx + c_idx * len(sensors)
                ax.plot(sample_scaled[:, _idx], label="orig norm")
                ax.plot(recon[:, _idx], c="red", linestyle="dashed", label="recon")
                ax.set_ylim([0, 1])
                if c_idx == len(converters) - 1:
                    ax.set_xlabel("time")
                if s_idx == 0:
                    ax.set_ylabel(converter, size='large')
                if c_idx == len(converters) - 1 and s_idx == 1:
                    ax.legend()

        fig.suptitle("Time window: %s, length=%i" % (np.datetime_as_string(time, unit="s"), sample.shape[0]))
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    @classmethod
    def plot_error_distribution(cls, errors, save_path=None, figsize=None, title="Error Distribution"):
        if figsize is None:
            fig = plt.figure(figsize=(2, 5))
        else:
            fig = plt.figure(figsize=figsize)
        plt.title(title)
        plt.boxplot(errors.reshape((1, -1)))
        plt.xticks([])
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    @classmethod
    def plot_recon_error(cls, errors, save_path=None):
        fig = plt.figure(figsize=(10, 3))
        plt.title("Reconstruction Error")
        plt.ylabel("MSE")
        plt.xlabel("time window idx")
        plt.scatter(list(range(errors.shape[0])), errors, alpha=1, s=1)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    @classmethod
    def plot_transformed_error_distribution(cls, errors, batch_errors, transformed_errors, title=None, save_path=None):
        color = plt.get_cmap("tab10").colors
        fig = cls.plot_error_distribution(errors, figsize=(3, 5))
        for err_b, err_t, c in zip(batch_errors, transformed_errors, color):
            x_pos = 3
            arrow = patches.FancyArrowPatch((x_pos, err_b.item()), (x_pos, err_t.item()),
                                            color=c,
                                            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                                            connectionstyle="arc3, rad=0.2")
            plt.scatter([x_pos, x_pos], [err_b.item(), err_t.item()], color=c, s=50)
            plt.gca().add_patch(arrow)
        plt.xlim(0, 5)
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        return fig
