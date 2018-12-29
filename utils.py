import numpy as np
from math import ceil


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input x

     """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    sig = sigmoid(x)
    return sig * (1 - sig)


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of xavier initialized np arrays weight matrices

    """
    return [xavier_initialization(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of zero np arrays weight matrices

    """
    return [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)]


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         np.array
             array of zero np arrays bias matrices

    """
    return [np.zeros((1, sizes[i])) for i in range(len(sizes))]


def create_batches(data, labels, size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    assert len(data) == len(labels)
    return [(data[i:i+size], labels[i:i+size]) for i in range(0, ceil(len(data)/size)*size, size)]


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    assert len(list1) == len(list2)
    return [list1[i] + list2[i] for i in range(len(list1))]


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


##########################################
# Main (some sanity checks)
##########################################

# if __name__ == "__main__":
#     import random
#
#     # just check that it doesn't fail
#
#     sizes = [random.randint(1,10) for _ in range(10)]
#     res = random_weights(sizes)
#     print(sizes)
#     for r in res:
#         print(r.shape)
#
#     print("=========================================")
#
#     sizes = [random.randint(1,10) for _ in range(10)]
#     res = zeros_weights(sizes)
#     print(sizes)
#     for r in res:
#         print(r.shape)
#     print(zeros_weights(sizes))
#
#     print("=========================================")
#
#     sizes = [random.randint(1,10) for _ in range(10)]
#     res = zeros_biases(sizes)
#     print(sizes)
#     for r in res:
#         print(r.shape)
#     print(zeros_biases(sizes))
#
#     print("=========================================")
#
#     data = [random.randint(1,10) for _ in range(5)]
#     labels = [random.randint(1,10) for _ in range(5)]
#     print(create_batches(data, labels, 3))
#
#     print("=========================================")
#
#     data = [random.randint(1,10) for _ in range(6)]
#     labels = [random.randint(1,10) for _ in range(6)]
#     print(create_batches(data, labels, 3))
#
#     print("=========================================")
#
#     lst1 = [x for x in range(6)]
#     lst2 = [x*2 for x in range(6)]
#     print(add_elementwise(lst1, lst2))
