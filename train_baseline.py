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
    opt_train = TrainOptions().parse()
    opt_val = TrainOptions().parse()

    # 设置显示验证结果存储的设置
    web_dir = os.path.join(opt_train.checkpoints_dir, opt_train.name, 'val')
    image_dir = os.path.join(web_dir, 'images')
    util.mkdirs([web_dir, image_dir])

    # 加载数据集
    dataset_train = create_dataset(opt_train)
    dataset_train_size = len(dataset_train)
    print('The number of training images = %d' % dataset_train_size)

    # 改变phase参数然后获取验证集(逻辑上有点问题，但是代码没有问题，因为singledataset中设置参数的静态方法没有执行)
    opt_val.phase = 'val'
    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)


    # 创建训练模型
    model_train = create_model(opt_train)
    model_train.train()
    #opt_train.continue_train = True
    model_train.setup(opt_train)

    # 创建验证模型
    opt_val.isTrain = False
    model_val = create_model(opt_val)
    model_val.eval()



    # 设置显示训练结果的类
    visualizer = Visualizer(opt_train)

    metrics = RunningScore(opt_train.num_classes)
    for epoch in range(opt_train.epoch_count, opt_train.niter + opt_train.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()
        metrics.reset()
        for i, data in enumerate(dataset_train):
            # label, image.shape == [N, C, W, H]
            iter_start_time = time.time()

            epoch_iters += 1

            # 训练一次
            model_train.set_input(data)
            model_train.optimize_parameters()

            # 保存训练出来的图像
            if epoch_iters % opt_train.display_freq == 0:
                visualizer.display_current_results_segment(model_train.get_current_visuals(), epoch)

            # 控制台打印loss的值，存储log信息到磁盘
            if epoch_iters % opt_train.print_freq == 0:
                losses = model_train.get_current_losses()
                t_comp = (time.time() - iter_start_time) / epoch_iters
                visualizer.print_current_losses(epoch, epoch_iters, losses, t_comp)

            gt = np.squeeze(data["B"].numpy(), axis = 1)  # [N, W, H]
            pre = model_train.pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            metrics.update(gt, pre)
        train_score, train_class_iou = metrics.get_scores()
        print("####################### train result start ##########################")
        print("Overall Acc:%.3f" % train_score["OverallAcc"] + " Mean Acc:%.3f" % train_score[
            "MeanAcc"] + " FreqW Acc:%.3f" % train_score["FreqWAcc"] + " Mean IoU:%.3f" % train_score[
                  "MeanIoU"] + " Class_IoU:" + str(train_class_iou))

        #if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d' % epoch)
        model_train.save_networks(epoch)



        # 在验证数据集上验证结果
        metrics.reset()
        # 从这一轮保存的权重恢复
        model_val.opt.epoch = epoch
        model_val.setup(model_val.opt)

        for i, data in enumerate(dataset_val):
            model_val.set_input(data)
            model_val.forward()
            gt = np.squeeze(data["B"].numpy(), axis = 1)  # [N, W, H]
            pre = model_val.pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            metrics.update(gt, pre)
            # 保存结果
            if i % opt_train.display_freq == 0:
                save_segment_result(model_val.get_current_visuals(), epoch, opt_train.display_winsize, image_dir, web_dir, opt_train.name)
        val_score, val_class_iou = metrics.get_scores()
        print("####################### val result start ##########################")
        print("Overall Acc:%.3f"%val_score["OverallAcc"] + " Mean Acc:%.3f"%val_score["MeanAcc"] + " FreqW Acc:%.3f"%val_score["FreqWAcc"] + " Mean IoU:%.3f"%val_score["MeanIoU"] + " Class_IoU:" + str(val_class_iou))
        # 一个epoch 改变一次学习率
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))


        model_train.update_learning_rate()

