import numpy as np


def convert_to_d100(prediction):
    return np.clip(round_multiple(prediction, multiple=10), a_min=10, a_max=100)


def round_multiple(array, multiple, dtype=None):
    if dtype is None:
        dtype = type(multiple)

    rounded_array = np.round(array / multiple) * multiple
    rounded_array = rounded_array.astype(dtype)

    return rounded_array
