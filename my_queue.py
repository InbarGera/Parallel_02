from multiprocessing import Lock, Pipe

# for debugging
from multiprocessing import Process
import time

class MyQueue(object):
    def __init__(self):
        self.input_end, self.output_end = Pipe(False)
        # self.readLock = Lock()
        self.writeLock = Lock()

    def put(self, msg):
        input, output = Pipe()
        output.send(msg)

        self.writeLock.acquire(True)
        self.output_end.send(input)
        self.writeLock.release()

    def get(self):
        next_pipe = self.input_end.recv()
        return next_pipe.recv()


# def threaded_function(queue, arg):
#     for i in range(arg):
#         queue.put("hello world")
#     return 0
#
# def reader_function(queue):
#     while True:
#         print(queue.get())
#
# if __name__ == '__main__':
#
#     threadlilngs = []
#     q = MyQueue()
#     threads_amount = 50
#     times_each_thread_prints = 1001
#
#     for _ in range(0, threads_amount):
#         threadlilngs.append(Process(target=threaded_function, args=(q, times_each_thread_prints)))
#
#     for i in range(0, threads_amount):
#         threadlilngs[i].start()
#     reader = Process(target=reader_function, args=(q,))
#     reader.start()
#     for i in range(0, threads_amount):
#         threadlilngs[i].join()
#     print("finish joining")
