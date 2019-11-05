from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_segment_result
from util.metrics import RunningScore
from util import util
import time
import os
import numpy as np

if __name__ == '__main__':
    # 加载设置
    opt = TrainOptions().parse()

    # 加载数据集
    dataset_train = create_dataset(opt)
    dataset_train_size = len(dataset_train)
    print('The number of training images = %d' % dataset_train_size)
    for i, data in enumerate(dataset_train):
        a = data
        stop = 1