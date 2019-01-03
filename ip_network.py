import multiprocessing as mp
import platform
import re
from network import *
from preprocessor import Worker

# from multiprocessing import Queue as ResultQueue
from my_queue import MyQueue as ResultQueue

# This takes long, let's make it async (doesn't help much though)
class SetJobsWorker(mp.Process):
    def __init__(self, n_consumers, data, labels, jobs_queue, n_jobs):
        super().__init__()

        self.n_consumers = n_consumers
        self.data = data
        self.labels = labels
        self.jobs_queue = jobs_queue
        self.n_jobs = n_jobs

    def run(self):
        for k in range(self.n_jobs):
            index = random.randrange(0, self.data.shape[0])
            self.jobs_queue.put((self.data[index], self.labels[index]))
        for i in range(self.n_consumers):  # get() pops an item, so we need a terminator for each worker
            self.jobs_queue.put((None, None))  # Indicates we're done (see Worker.run())


class IPNeuralNetwork(NeuralNetwork):
    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16, epochs=10, matmul=np.matmul):
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)

        self.workers = []
        self.jobs = mp.Queue()       # Queue of tuples (image, label)
        self.results = ResultQueue() # Queue of tuples (image, label)

    def _n_cpus(self):
        if platform.system() == 'Windows':
            return mp.cpu_count()  # Good for tests, but gets wrong number on CDP servers
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', open('/proc/self/status').read())
        num_cpu = bin(int(m.group(1).replace(',', ''), 16)).count('1')
        return num_cpu

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        #Create Workers and set jobs
        n_workers = self._n_cpus()

        data = training_data[0]
        labels = training_data[1]
        jobs_worker = SetJobsWorker(n_workers, data, labels, self.jobs,
                                    n_jobs=(self.number_of_batches * self.mini_batch_size * self.epochs))
        jobs_worker.start()

        for _ in range(n_workers):
            worker = Worker(self.jobs, self.results)
            worker.start()
            self.workers.append(worker)

        #Call the parent's fit 
        super().fit(training_data, validation_data)
        
        #Stop Workers
        for worker in self.workers:
            worker.join()
        self.workers = []
        jobs_worker.join()

    def create_batches(self, data, labels, batch_size):
        """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch
    
         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

        """
        batches_flat = []  # all augmented data in one list, without splitting into batches
        for k in range(self.number_of_batches * self.mini_batch_size):
            # Stop condition for results queue:
            #  we know that number of results is the same as number of jobs
            #  so here we don't use None-terminated queue like with jobs
            batches_flat.append(self.results.get())

        batches = []
        sz = self.mini_batch_size
        for i in range(self.number_of_batches):
            batch = batches_flat[i * sz : (i + 1) * sz]  # list of tuples (image, label)
            batches.append((np.array([tup[0] for tup in batch]), np.array([tup[1] for tup in batch])))  # tuple of (data batch of batch_size, labels batch of batch_size)
        return batches


    
