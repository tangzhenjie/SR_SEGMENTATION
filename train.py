from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time

if __name__ == '__main__':
    # 加载设置
    opt = TrainOptions().parse()

    # 加载数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # 创建模型
    model = create_model(opt)

    # 设置学习率，和恢复权重
    model.setup(opt)

    # 设置显示训练结果的类
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters += 1
            epoch_iters += 1

            # 训练一次
            model.set_input(data)
            model.optimize_parameters()

            # 保存训练出来的图像
            if total_iters % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            # 控制台打印loss的值，存储log信息到磁盘
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = time.time() - iter_start_time
                visualizer.print_current_losses(epoch, epoch_iters, losses, t_comp)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        # 一个epoch 改变一次学习率
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

