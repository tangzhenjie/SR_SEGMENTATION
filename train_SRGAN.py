from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time

if __name__ == '__main__':
    # 加载设置
    opt = TrainOptions().parse()
    opt.upscale_factor = 4
    # 加载训练数据集
    dataset_train = create_dataset(opt)
    dataset_train_size = len(dataset_train)
    print('The number of training images = %d' % dataset_train_size)

    # 加载验证数据集
    opt.phase = "val"
    opt.batch_size = 1
    opt.serial_batches = True
    dataset_val = create_dataset(opt)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()
        for i, data in enumerate(dataset_train):
            iter_start_time = time.time()



