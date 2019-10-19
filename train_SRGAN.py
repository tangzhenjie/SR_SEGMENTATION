from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util import util
from math import log10
from util.visualizer import save_sr_result
import time
import os
import pytorch_ssim


if __name__ == '__main__':
    # 加载设置
    opt = TrainOptions().parse()
    opt.upscale_factor = 4

    # 设置显示验证结果存储的设置
    web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val')
    image_dir = os.path.join(web_dir, 'images')
    util.mkdirs([web_dir, image_dir])


    # 加载训练数据集
    dataset_train = create_dataset(opt)
    dataset_train_size = len(dataset_train)
    print('The number of training images = %d' % dataset_train_size)

    # 加载验证数据集
    opt1 = TrainOptions().parse()
    opt1.upscale_factor = 4
    opt1.phase = "val"
    opt1.batch_size = 1
    opt1.serial_batches = True
    dataset_val = create_dataset(opt1)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    # 构建model
    model = create_model(opt)

    # 设置学习率，和恢复权重
    model.setup(opt)

    # 设置显示训练结果的类
    visualizer = Visualizer(opt)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()
        model.train()
        for i, data in enumerate(dataset_train):
            iter_start_time = time.time()
            epoch_iters += 1

            # 训练一次
            model.set_input(data)
            model.optimize_parameters()

            # 保存训练出来的图像
            if epoch_iters % opt.display_freq == 0:
                visualizer.display_current_results_sr(model.get_current_visuals(), epoch)

            # 控制台打印loss的值，存储log信息到磁盘
            if epoch_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / epoch_iters
                visualizer.print_current_losses(epoch, epoch_iters, losses, t_comp)
                # 在验证数据集上验证结果

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % epoch)
            model.save_networks(epoch)
        model.eval()
        batch_sizes = 0
        mse = 0
        ssims = 0
        psnr = 0
        psnr = 0
        ssim = 0
        for i, data in enumerate(dataset_val): # batchsize = 1
            batch_sizes += 1 # because batch_size = 1
            model.set_input(data)
            model.forward()
            sr = model.sr_img
            hr = model.hr_image
            batch_mse = ((sr - hr) ** 2).data.mean()
            batch_ssim = pytorch_ssim.ssim(sr, hr).data
            mse += batch_mse * 1  # because batch_size = 1
            ssims += batch_ssim * 1 # because batch_size = 1
            psnr = 10 * log10(1 / (mse / batch_sizes))
            ssim = ssims / batch_sizes

            # 保存结果
            if i % opt.display_freq == 0:
                save_sr_result(model.get_current_visuals(), epoch, opt.display_winsize, image_dir, web_dir,
                                    opt.name)
        print("[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f" %(psnr, ssim))
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()




