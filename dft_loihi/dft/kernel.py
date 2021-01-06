from abc import ABC, abstractmethod
import numpy as np
from dft_loihi.dft.util import gauss, shift_fill


class Kernel(ABC):
    def __init__(self):
        self.weights = None
        self.mask = None

    def compute_mask(self, weights):
        return np.where(weights == 0, 0, 1)

    def compute_weights(self, kernel_slice, field_shape):
        ndims = len(field_shape)

        if ndims == 1:
            field_size = field_shape[0]

            size_diff = field_size - len(kernel_slice)

            if size_diff != 0:
                rest = int(np.floor(np.abs(size_diff) / 2.0))

                if size_diff > 0:
                    pad_width = [rest] * 2

                    if size_diff % 2 != 0:
                        pad_width[0] = pad_width[0] + 1

                    kernel_slice = np.pad(kernel_slice,
                                          pad_width,
                                          'constant',
                                          constant_values=0)
                elif size_diff < 0:
                    kernel_slice = kernel_slice[rest:rest+field_size]
                    print("rest: " + str(rest))
                    print("kernel slice: " + str(kernel_slice))

            weights = np.zeros((field_size, field_size))

            for i in range(field_size):
                shift = i - int(np.floor(field_size/2.0))

                weights[i, :] = self.shift_kernel_slice(kernel_slice, shift)
                if self.border_type[0] == "circular":
                    weights[i, :] = np.roll(kernel_slice, shift, axis=0)
                elif self.border_type[0] == "zeros":
                    weights[i, :] = shift_fill(kernel_slice, shift, fill_value=0)
                else:
                    weights[i, :] = self.shift_kernel_slice(kernel_slice, shift)


        else:
            raise ValueError("Error: Kernel not implemented for specified dimensionality.")

        return weights

    @abstractmethod
    def shift_kernel_slice(self, kernel_slice, shift):
        pass


class SelectiveKernel(Kernel):
    """A kernel that enables creating a selective dynamic neural field
    (local excitation, global inhibition)."""

    def __init__(self,
                 amp_exc=1.0,
                 width_exc=1.0,
                 center_exc=0.0,
                 global_inh=0.1,
                 border_type="inhibition"):
        super().__init__()

        if type(amp_exc) == float or type(amp_exc) == int:
            amp_exc = (amp_exc,)
            width_exc = np.array([width_exc], dtype=np.float32)
            center_exc = np.array([center_exc], dtype=np.float32)
            border_type = [border_type]

        self.amp_exc = amp_exc
        self.width_exc = width_exc
        self.center_exc = center_exc
        self.global_inh = global_inh if global_inh < 0 else -1 * global_inh
        self.border_type = border_type

    def estimate_domain_shape(self, field_domain, field_shape, center, width):
        sampling = (field_domain[:, 1] - field_domain[:, 0]) / field_shape[:]
        # estimate the shape of the kernel
        kernel_shape = 2 * np.array(field_shape)
        # ensure that the kernel has an odd size
        kernel_shape = np.where(kernel_shape % 2 == 0, kernel_shape + 1, kernel_shape)

        # compute the domain of the kernel
        kernel_domain = np.zeros(field_domain.shape)
        half_domain = kernel_shape * sampling / 2.0
        kernel_domain[:, 0] = center - half_domain
        kernel_domain[:, 1] = center + half_domain

        return kernel_domain, tuple(kernel_shape)

    def create(self, field_domain, field_shape):
        # compute domain and shape of the excitatory kernel
        domain_exc, shape_exc = self.estimate_domain_shape(field_domain,
                                                           field_shape,
                                                           self.center_exc,
                                                           self.width_exc)

        print("domain exc: " + str(domain_exc))
        print("shape exc: " + str(shape_exc))

        local_excitation = gauss(domain_exc,
                                 shape_exc,
                                 self.amp_exc,
                                 self.center_exc,
                                 self.width_exc)

        kernel_slice = local_excitation + self.global_inh

        self.weights = self.compute_weights(kernel_slice, field_shape)
        self.mask = self.compute_mask(self.weights)

    def shift_kernel_slice(self, kernel_slice, shift):
        if self.border_type[0] == "inhibition":
            return shift_fill(kernel_slice, shift, fill_value=self.global_inh)
        else:
            raise ValueError("Border type not implemented.")


class MultiPeakKernel(Kernel):
    """A class that enables configuring a "Mexican hat" kernel
    (local excitation and mid-range inhibition) that enables
    multiple peaks within a field."""

    def __init__(self,
                 amp_exc=1.0,
                 width_exc=1.0,
                 center_exc=0.0,
                 amp_inh=-1.0,
                 width_inh=2.0,
                 center_inh=0.0,
                 limit=1.0,
                 border_type="zero"):
        super().__init__()

        if type(amp_exc) == float or type(amp_exc) == int:
            amp_exc = (amp_exc,)
            amp_inh = (amp_inh,)
            width_exc = np.array([width_exc], dtype=np.float32)
            width_inh = np.array([width_inh], dtype=np.float32)
            center_exc = np.array([center_exc], dtype=np.float32)
            center_inh = np.array([center_inh], dtype=np.float32)
            border_type = [border_type]

        self.amp_exc = amp_exc
        self.width_exc = width_exc
        self.center_exc = center_exc
        self.amp_inh = amp_inh
        self.width_inh = width_inh
        self.center_inh = center_inh
        self.limit = limit
        self.border_type = border_type

        assert len(self.width_exc) == len(self.width_inh) and \
               len(self.center_exc) == len(self.center_inh), \
               "Excitatory and inhibitory kernel need to have the same dimensionality."

    def estimate_domain_shape(self, field_domain, field_shape, center, width):
        sampling = (field_domain[:, 1] - field_domain[:, 0]) / field_shape[:]
        # estimate the shape of the kernel
        kernel_shape = np.uint(np.ceil(2 * self.limit * width / sampling))
        # ensure that the kernel has an odd size
        kernel_shape = np.where(kernel_shape % 2 == 0, kernel_shape + 1, kernel_shape)

        # compute the domain of the kernel
        kernel_domain = np.zeros(field_domain.shape)
        half_domain = kernel_shape * sampling / 2.0
        kernel_domain[:, 0] = center - half_domain
        kernel_domain[:, 1] = center + half_domain

        return kernel_domain, tuple(kernel_shape)

    def create(self, field_domain, field_shape):
        # compute domain and shape of the excitatory kernel
        domain_exc, shape_exc = self.estimate_domain_shape(field_domain,
                                                           field_shape,
                                                           self.center_exc,
                                                           self.width_exc)

        print("domain exc: " + str(domain_exc))
        print("shape exc: " + str(shape_exc))

        local_excitation = gauss(domain_exc,
                                 shape_exc,
                                 self.amp_exc,
                                 self.center_exc,
                                 self.width_exc)

        print("local excitation: " + str(local_excitation))

        domain_inh, shape_inh = self.estimate_domain_shape(field_domain,
                                                           field_shape,
                                                           self.center_inh,
                                                           self.width_inh)

        print("domain inh: " + str(domain_inh))
        print("shape inh: " + str(shape_inh))

        mid_range_inhibition = gauss(domain_inh,
                                     shape_inh,
                                     self.amp_inh,
                                     self.center_inh,
                                     self.width_inh)

        print("mid range inhibition: " + str(mid_range_inhibition))

        # pad the smaller array with zeros so that they have the same size
        shape_diff = np.array(mid_range_inhibition.shape) - np.array(local_excitation.shape)

        print("shape diff: " + str(shape_diff))
        for i, diff in enumerate(shape_diff):
            if diff > 0:
                pad_width = np.zeros((local_excitation.ndim,) + (2,), dtype=np.int32)
                pad_width[i][:] = np.ceil(diff / 2)
                local_excitation = np.pad(local_excitation,
                                          pad_width,
                                          'constant',
                                          constant_values=0)
            if diff < 0:
                pad_width = np.zeros((mid_range_inhibition.ndim,) + (2,), dtype=np.int32)
                pad_width[i][:] = np.ceil(diff / 2)
                mid_range_inhibition = np.pad(mid_range_inhibition,
                                              pad_width,
                                              'constant',
                                              constant_values=0)

        kernel_slice = local_excitation + mid_range_inhibition
        print("kernel slice: " + str(kernel_slice))

        self.weights = self.compute_weights(kernel_slice, field_shape)
        self.mask = self.compute_mask(self.weights)

    def shift_kernel_slice(self, kernel_slice, shift):
        raise ValueError("Border type not implemented.")
