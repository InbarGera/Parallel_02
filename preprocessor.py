import multiprocessing
import numpy as np
import random
from scipy import ndimage

class Worker(multiprocessing.Process):
    def __init__(self, jobs, result, training_data=None, batch_size=None):
        super().__init__()
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: Queue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs       # Queue of tuples (image, label)
        self.results = result  # Queue of tuples (image, label)

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        return ndimage.rotate(image, angle, reshape=False)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        res = np.roll(image, -dx, axis=1)
        res = np.roll(res, -dy, axis=0)
        # ugly, but I don't care
        if dy > 0:
            res[-dy:, :] = 0
        else:
            res[:-dy, :] = 0
        if dx > 0:
            res[:, -dx:] = 0
        else:
            res[:, :-dx] = 0
        return res
        # this would be so much simpler, but 'You may only use numpy library in the function'
        # return ndimage.shift(image, [-dx, -dy])
    
    @staticmethod
    def step_func(image, steps):
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''
        # we normalize 0-255 pixels to [0,1] range, then apply step function, then rescale it back to 0-255
        return ((np.floor(steps * (image / 255)) / (steps - 1)) * 255).astype(int)

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        res = np.zeros_like(image)
        for y, x in np.ndindex(image.shape):
            x_src = int(x + y * tilt)
            if x_src < 0 or x_src >= image.shape[1]:
                continue
            res[y, x] = image[y, x_src]
        return res

    def process_image(self, image):
        '''Apply the image process functions

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        # Be careful with those, too large values can make image unrecognizable
        skew = 0.1
        rotate = 15
        shift = 2

        image2d = image.reshape((28, 28))  # TODO: hardcoded :(

        res = image2d  # so we can freely reorder below lines
        res = Worker.skew(res, random.uniform(-skew, skew))
        res = Worker.rotate(res, random.randint(-rotate, rotate))
        res = Worker.shift(res, random.randint(-shift, shift), random.randint(-shift, shift))
        # res = Worker.step_func(res, random.randint(16, 255)) # This only makes it worse

        res = res.reshape((28*28,))
        return res

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
        '''
        # This is consumer of jobs, and producer of results
        while True:
            img, label = self.jobs.get()
            if img is None: # TODO: need to explicitly put(None) into queue when finished, to indicate that nothing more will be put
                return
            res = self.process_image(img)
            self.results.put((res, label))


##########################################
# Main (some sanity checks)
##########################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_img = ndimage.imread('6.png', flatten=True)

    # plt.imshow(Worker.rotate(test_img, 30))
    # plt.show()
	#
    # X = Worker.shift(test_img, -50, -100)
    # plt.imshow(X)
    # plt.show()
    #
    # X = Worker.shift(test_img, 50, 100)
    # plt.imshow(X)
    # plt.show()
	#
    # Y = Worker.step_func(test_img, 4)
    # plt.imshow(Y)
    # plt.show()
	#
    # Y = Worker.skew(test_img, 0.3)
    # plt.imshow(Y)
    # plt.show()

    worker = Worker(multiprocessing.Queue(), multiprocessing.Queue())

    for _ in range(10):
        Img = worker.process_image(test_img)
        Img = Img.reshape((28,28))
        plt.imshow(Img)
        plt.show()
