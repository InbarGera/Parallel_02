from numba import cuda
from numba import njit
import numpy as np


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
    kernel_size = (kernel.shape[0], kernel.shape[1])
    new_kernel = np.zeros(kernel_size)

    x_middle = (int)(kernel.shape[0] / 2 - (kernel.shape[0] / 2) % 1)  # got lazy to include math lib, this is simply floor
    y_middle = (int)(kernel.shape[1] / 2 - (kernel.shape[1] / 2) % 1)

    # first fliping the kernel martix
    for x in range(0, kernel.shape[0]):
        x_diff = abs(x - x_middle)*2
        if (x > x_middle):
            new_x = x - x_diff
        else:
            new_x = x + x_diff
        for y in range(0, kernel.shape[1]):
            y_diff = abs(y - y_middle)*2
            if (y > y_middle):
                new_y = y - y_diff
            else:
                new_y = y + y_diff
            new_kernel[x][y] = kernel[new_x][new_y]

    kernel = new_kernel # wrote the latter code with kernel, didnt want to change it

    rows = image.shape[0]
    cols = image.shape[1]
    size = (rows, cols)
    res = np.zeros(size)
    for i in range(0, rows):
        for j in range(0, cols):
            temp = 0
            for x in range(0, kernel.shape[0]):
                for y in range(0, kernel.shape[1]):
                    a = i + x - x_middle
                    b = j + y - y_middle
                    if(a < rows and a >= 0 and b < cols and b >= 0):
                            temp += image[a][b] * kernel[x][y]
            res[i][j] = temp
    return res

@cuda.jit
def convolution_kernel():
    pass  # TODO