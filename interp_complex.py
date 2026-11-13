import numpy as np
from scipy.ndimage import zoom


def interp_complex_nd(arr, new_shape_2d, order=1):
    """
    Interpolate the last two dimensions of a complex array to a new shape, broadcasting over others.

    Parameters
    ----------
    arr : np.ndarray
        Input complex array with shape (..., H, W)
    new_shape_2d : tuple of int
        Target shape for the last two dimensions (new_H, new_W)
    order : int, optional
        Spline interpolation order (0-5), default is 1 (linear).

    Returns
    -------
    np.ndarray
        Interpolated complex array with shape (..., new_H, new_W)
    """
    # Compute zoom factors for last two dims
    zoom_factors = [1] * (arr.ndim - 2) + [new_shape_2d[0] / arr.shape[-2],
                                          new_shape_2d[1] / arr.shape[-1]]
    
    # Interpolate real and imaginary parts separately
    real_resized = zoom(arr.real, zoom_factors, order=order)
    imag_resized = zoom(arr.imag, zoom_factors, order=order)
    
    return real_resized + 1j * imag_resized


def interp_complex_nd_mem(arr, new_shape_2d, order=1):
    """
    Memory-efficient interpolation of the last two dimensions of a complex array.

    Parameters
    ----------
    arr : np.ndarray
        Input complex array with shape (..., H, W)
    new_shape_2d : tuple of int
        Target shape for the last two dimensions (new_H, new_W)
    order : int, optional
        Spline interpolation order (0-5), default is 1 (linear).

    Returns
    -------
    np.ndarray
        Interpolated complex array with shape (..., new_H, new_W)
    """
    leading_shape = arr.shape[:-2]
    new_arr_shape = leading_shape + new_shape_2d
    result = np.empty(new_arr_shape, dtype=arr.dtype)

    zoom_factors = (new_shape_2d[0] / arr.shape[-2], new_shape_2d[1] / arr.shape[-1])

    # Iterate over all indices in the leading dimensions
    it = np.ndindex(leading_shape)
    for idx in it:
        result[idx] = zoom(arr[idx].real, zoom_factors, order=order) + 1j * zoom(arr[idx].imag, zoom_factors, order=order)

    return result
