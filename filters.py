from numba import cuda
from numba import njit
import numpy as np
import math


@njit
def flip_kernel(kernel):
    kernel_size = (kernel.shape[0], kernel.shape[1])
    new_kernel = np.zeros(kernel_size)

    x_middle = math.floor(kernel.shape[0]/2)
    y_middle = math.floor(kernel.shape[1]/2)

    for x in range(0, kernel.shape[0]):
        x_diff = abs(x - x_middle) * 2
        if (x > x_middle):
            new_x = x - x_diff
        else:
            new_x = x + x_diff
        for y in range(0, kernel.shape[1]):
            y_diff = abs(y - y_middle) * 2
            if (y > y_middle):
                new_y = y - y_diff
            else:
                new_y = y + y_diff
            new_kernel[x][y] = kernel[new_x][new_y]
    return new_kernel


# contstants for the rest of the file
threads_per_block = 1024
t = 32
blocks = 16
block_line = 4


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
    kernel = flip_kernel(kernel)

    # convolution itself
    x_middle = math.floor(kernel.shape[0] / 2)
    y_middle = math.floor(kernel.shape[1] / 2)

    rows = image.shape[0]
    cols = image.shape[1]
    size = (rows, cols)
    res = np.zeros(size)
    # for every entry in image
    for i in range(0, rows):
        for j in range(0, cols):
            cell_value = 0
            # calculate the convolution in this cell
            for x in range(0, kernel.shape[0]):
                for y in range(0, kernel.shape[1]):
                    a = i + x - x_middle
                    b = j + y - y_middle
                    # making sure the neighbours are in the image boundaries
                    if(a < rows and a >= 0 and b < cols and b >= 0):
                        cell_value += image[a][b] * kernel[x][y]
            res[i][j] = cell_value
    return res


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

    kernel = flip_kernel(kernel)

    delta_y_block = math.ceil(image.shape[0] / block_line)
    delta_x_block = math.ceil(image.shape[1] / block_line)

    Z = np.zeros((image.shape[0], image.shape[1]))

    gpu_arr = cuda.to_device(Z)
    convolution_kernel[blocks, threads_per_block](kernel, image, delta_x_block, delta_y_block, gpu_arr)
    res = gpu_arr.copy_to_host()

    return res

@cuda.jit
def convolution_kernel(kernel, image, delta_x_block, delta_y_block, res):

    thread_num = cuda.threadIdx.x
    block_num = cuda.blockIdx.x

    # block offset in the general image
    x_block_offset = delta_x_block * (block_num % block_line)
    y_block_offset = delta_y_block * (int)((block_num/block_line - (block_num/block_line % 1.0)))

    # thread offset in the block
    # t = sqrt(threads_per_block), done outside because the gpu is running out of resources otherwise
    x_per_thread = (int)((delta_x_block / t) - ((delta_x_block / t) % 1.0)) + 1
    y_per_thread = (int)((delta_y_block / t) - ((delta_y_block / t) % 1.0)) + 1

    x_thread_offset = (thread_num % t) * x_per_thread
    y_thread_offset = ((thread_num/t) - ((thread_num/t) % 1)) * y_per_thread

    # total offset for thread
    x_offset = (int)(x_block_offset + x_thread_offset)
    y_offset = (int)(y_block_offset + y_thread_offset)

    # convolution itself

    # dont know how cuda will handle math lib functions, so implements floor manually
    x_middle = (int)(kernel.shape[0] / 2 - (kernel.shape[0] / 2) % 1)
    y_middle = (int)(kernel.shape[1] / 2 - (kernel.shape[1] / 2) % 1)

    rows = image.shape[0]
    cols = image.shape[1]

    for i in range(0, y_per_thread):
        i += y_offset
        # checking that i is in boundaries of table, and block authority
        if(i < rows and i >= 0 and (i >= y_block_offset and i < y_block_offset + delta_y_block)):
            for j in range(0, x_per_thread):
                j += x_offset
                # same check for j
                if((j < cols and j >= 0) and (j >= x_block_offset and j < x_block_offset + delta_x_block)):
                    # convolution itself
                    cell_value = 0
                    for x in range(0, kernel.shape[0]):
                        for y in range(0, kernel.shape[1]):
                            a = i + x - x_middle
                            b = j + y - y_middle
                            if (a < rows and a >= 0 and b < cols and b >= 0):
                                cell_value += image[a][b] * kernel[x][y]
                    res[i][j] = cell_value




