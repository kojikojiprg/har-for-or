import io

import numpy as np


def images_to_tensor(npy, transform):
    npy = np.lib.format.read_array(io.BytesIO(npy))
    return transform(npy)
