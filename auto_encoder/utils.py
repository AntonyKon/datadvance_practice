import numpy as np
from .builder import Builder
from da.p7core_6_23_400 import gtapprox
import category_encoders as ce
import matplotlib.pyplot as plt
import re
from datetime import timedelta


def convert_to_timedelta(timedelta_str):
    pattern = re.compile(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(\.(?P<microseconds>\d+))?')
    m = pattern.match(timedelta_str)

    params = {key: float(val) for key, val in m.groupdict(0).items()}

    return timedelta(**params)


def plot(default_model_RRMS, encoded_model_RRMS, default_model_time, encoded_model_time):
    fig, axes = plt.subplots(2, 1, figsize=(10, 20))

    errors = np.unique(np.concatenate([default_model_RRMS, encoded_model_RRMS]))
    times = np.unique(np.concatenate([default_model_time, encoded_model_time]))

    encoded_model_count = np.array([np.count_nonzero(encoded_model_RRMS < error) for error in errors])
    default_model_count = np.array([np.count_nonzero(default_model_RRMS < error) for error in errors])

    encoded_model_time_count = np.array([np.count_nonzero(encoded_model_time < time) for time in times])
    default_model_time_count = np.array([np.count_nonzero(default_model_time < time) for time in times])

    fontsize = 20

    axes[0].plot(errors, default_model_count, label="gtapprox.Model")
    axes[0].plot(errors, encoded_model_count, label="Model with encoding")
    axes[0].set_xlabel('RRMS error', fontsize=fontsize)
    axes[0].set_ylabel('Model number', fontsize=fontsize)

    axes[1].plot(times, default_model_time_count, label="gtapprox.Model")
    axes[1].plot(times, encoded_model_time_count, label="Model with encoding")
    axes[1].set_xlabel('Training time', fontsize=fontsize)
    axes[1].set_ylabel('Model number', fontsize=fontsize)

    for ax in axes:
        ax.grid()
        ax.legend(fontsize=fontsize)

    plt.show()
    fig.savefig("result.png")
