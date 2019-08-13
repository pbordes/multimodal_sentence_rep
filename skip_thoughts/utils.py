import os

def select_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return
