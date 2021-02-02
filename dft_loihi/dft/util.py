import numpy as np
import scipy.stats
from lava.processes.n3.connections import Connections


ms_per_time_step = 10
time_steps_per_ms = 1 / ms_per_time_step
time_steps_per_second = 1000 * time_steps_per_ms
time_steps_per_minute = 60 * time_steps_per_second


def tau_to_decay(tau):
    return int(4095 / tau)


def connect(source, target, weights, mask="one-to-one"):

    if isinstance(weights, int) or isinstance(weights, float):
        weights = np.asarray([weights])
    elif isinstance(weights, list):
        weights = np.asarray(weights)

    if type(mask) == str:
        if mask == "full":
            mask = np.full((source.size, target.size), 1.0)
        elif mask == "one-to-one":
            mask = np.eye(source.size, target.size)

    connections = Connections(name='conn',
                              conn_mask=mask,
                              wgts=weights,
                              sign_mode=2)

    connections(s_pre=source.s_out)
    target(a_in=connections.a)


# preallocate empty array and assign slice by chrisaycock
def shift_fill(array, shift, fill_value=0):
    result = np.empty_like(array)
    if shift > 0:
        result[:shift] = fill_value
        result[shift:] = array[:-shift]
    elif shift < 0:
        result[shift:] = fill_value
        result[:shift] = array[-shift:]
    else:
        result[:] = array
    return result


def gauss(domain, shape, amplitude=1.0, mean=None, stddev=None):
    if type(shape) == int:
        domain = np.array([domain], dtype=np.float32)
        shape = (shape,)
        stddev = [stddev]

    ndim = len(shape)

    # Mean is zero by default
    if mean is not None:
        mean = mean
    else:
        mean = np.zeros(ndim, dtype=np.float32)

    # Stddev is 1 in each dimension by default
    if stddev is None:
        stddev = np.ones(ndim, dtype=np.float32)

    # Assemble a set of linear spaces
    linspaces = []
    for i in range(0, ndim):
        linspaces.append(np.linspace(domain[i][0], domain[i][1], shape[i], endpoint=True, dtype=np.float32))
    linspaces = np.array(linspaces)

    # Combine linear spaces into meshgrid
    mgrid = np.array(np.meshgrid(*linspaces))

    # Reshape meshgrid
    pos = np.zeros(shape + (len(domain),), dtype=np.float32)
    for i in range(0, ndim):
        if ndim == 1:
            pos[:, i] = mgrid[i, :]
        elif ndim == 2:
            pos[:, :, i] = mgrid[i, :, :]
        elif ndim == 3:
            pos[:, :, :, i] = mgrid[i, :, :, :]
        elif ndim == 4:
            pos[:, :, :, :, i] = mgrid[i, :, :, :, :]

    # Define multivariate distribution over meshgrid
    unnormalized = scipy.stats.multivariate_normal.pdf(
        pos,
        mean=mean,
        cov=stddev)

    normalized = unnormalized / np.max(unnormalized)
    result = normalized * amplitude

    return result

