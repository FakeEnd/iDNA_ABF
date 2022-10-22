import torch
import pynvml

'''
This file checks the states of GPU devices.
'''


def gpus():
    print('GPU is_available:', torch.cuda.is_available())

    pynvml.nvmlInit()
    gpu_num = pynvml.nvmlDeviceGetCount()
    print('gpu num:', gpu_num)

    for i in range(gpu_num):
        print('-' * 50, 'gpu[{}]'.format(str(i)), '-' * 50)
        gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
        print('gpu object:', gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
        print('total memory:', meminfo.total / 1024 ** 3, 'GB')
        print('using memory:', meminfo.used / 1024 ** 3, 'GB')
        print('remaining memory:', meminfo.free / 1024 ** 3, 'GB')


if __name__ == '__main__':
    gpus()
