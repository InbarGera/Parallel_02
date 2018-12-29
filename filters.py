from numba import cuda
from numba import njit


def convolution_gpu(kernel, image):
    '''Convolve using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


@njit
def convolution_numba(kernel, image):
    '''Convolve using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


@cuda.jit
def convolution_kernel():
    pass  # TODO