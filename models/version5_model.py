import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.loss as loss

class Version5Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for determining the class number')
        if is_train:
            parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='the type of GAN objective.')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G", "D"]    # "loss_"
        self.visual_names = ["imagelrB", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imagelrA", "imagesrA", "imagesrA_down",
                                  "imageA", "imagesrB", "imageB",
                                  "pixelfakeA_out", "pixelfakeB_out", "pre",  "pre_B"]

        self.model_names = ['generator']  # "net"
        if self.isTrain:
            self.model_names += ['pixel_discriminator', 'fc_discriminator']


        self.netgenerator = networks.generator(num_cls=opt.num_classes, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # 像素空间判别器
            self.netpixel_discriminator = networks.define_D(3, 64, 'basic', gpu_ids=self.gpu_ids)

            # 输出空间判别器
            self.netfc_discriminator = networks.fc_discriminator(gpu_ids=self.gpu_ids)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 像素判别器损失
            self.mse_loss = networks.GANLoss("lsgan").to(self.device)

            # 输出空间判别器损失
            self.bce_loss = networks.GANLoss("vanilla").to(self.device)

            # 超分辨损失函数
            self.generator_criterion = networks.GeneratorLoss().to(self.device)

            # 内容一致损失
            self.L2_loss = nn.MSELoss().to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netgenerator.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netpixel_discriminator.parameters(),
                                                                self.netfc_discriminator.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            self.imagelrA = input["A"].to(self.device)  # [3, 60, 60] 低分辨率图像
            self.label = input["B"].to(self.device)  # [1, 240, 240] 低分辨率图像对应的上采样label图
            self.imagelrB = input["C"].to(self.device)  # [3, 60, 60] 高分的biculic下采样图
            self.imageB = input["D"].to(self.device)  # [3, 240, 240] 高分图像
            self.imageA = input["E"].to(self.device)  # [3, 240, 240] 高分A图像 72 -> 240
        else:
            self.imagelrB = input["A"].to(self.device)  # [3, 60, 60] 高分图像
            self.label = input["B"].to(self.device)  # [1, 240, 240] 高分label图


    def forward(self):
        if self.isTrain:
            ##########################
            # imagelrA segmentation: imagelrA[60, 60] => label[240, 240] 注意这里面还有一个imagesrA 没有使用
            ##########################

            # iamgelrA 通过 generator
            self.feature_A, _, self.imagesrA, self.imagesrA_down = self.netgenerator(self.imagelrA)
            self.imagesrA_down_cut = self.imagesrA_down.detach() # 隔断反向传播
            self.feature_imagesrA_down, self.pre, _, _ = self.netgenerator(self.imagesrA_down_cut)
            self.pre_A_cut = self.pre.detach()  # 隔断反向传播

            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)
            self.imagesrA_cut = self.imagesrA.detach() # 隔断反向传播


            #imagelrB 通过 generator
            _, self.pre_B, self.imagesrB, _ = self.netgenerator(self.imagelrB)
            self.imagesrB_cut = self.imagesrB.detach() # 隔断反向传播
            self.pre_B_cut = self.pre_B.detach()  # 隔断反向传播

            # imagesrA 通过判别器
            self.pixelfakeA_out = (F.tanh(self.netpixel_discriminator(self.imagesrA)) + 1) * 0.5
            self.pixelfakeB_out = (F.tanh(self.netpixel_discriminator(self.imagesrB)) + 1) * 0.5

            self.fcreal_out = self.netfc_discriminator(F.softmax(self.pre_B, dim=1))

        else:
            _, self.pre, _, _ = self.netgenerator(self.imagelrB)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算两个损失"""

        # 分割损失
        self.loss_cross_entropy = self.loss_function(self.pre, self.label.long().squeeze(1))

        # 像素级对齐损失
        self.loss_da1 = self.mse_loss(self.pixelfakeA_out, True)

        # 超分辨GAN对齐
        self.loss_da2 = self.mse_loss(self.pixelfakeB_out, True)

        # 输出空间对齐
        self.loss_da3 = self.bce_loss(self.fcreal_out, False)

        # 超分辨损失
        self.loss_sr = self.generator_criterion(self.imagesrB, self.imageB, is_sr=True)

        # A内容一致性损失
        self.loss_idtA = self.generator_criterion(self.imagesrA, self.imageA, is_sr=False)

        # fix_pointA loss
        self.loss_fix_point = self.L2_loss(self.feature_A, self.feature_imagesrA_down)

        loss_DA = self.loss_da1 + self.loss_da2 + self.loss_da3
        loss_ID = self.loss_sr + self.loss_idtA + self.loss_cross_entropy

        # 求分割损失和超分辨损失的和
        self.loss_G =  loss_DA  * 4 + loss_ID * 10 + self.loss_fix_point * 8
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):

        pixeltrueB_out = (F.tanh(self.netpixel_discriminator(self.imageB)) + 1) * 0.5
        # 域转换判别器损失(F.tanh(self.netpixel_discriminator(self.imagesrB_cut)) + 1) * 0.5
        self.loss_D_da1 = self.mse_loss((F.tanh(self.netpixel_discriminator(self.imagesrA_cut)) + 1) * 0.5, False) \
                          + self.mse_loss(pixeltrueB_out, True)
        self.loss_D_da2 = self.mse_loss((F.tanh(self.netpixel_discriminator(self.imagesrB_cut)) + 1) * 0.5, False) \
                          + self.mse_loss(pixeltrueB_out, True)
        self.loss_D_da3 = self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_A_cut, dim=1)), False) \
                          + self.bce_loss(self.netfc_discriminator(F.softmax(self.pre_B_cut, dim=1)), True)
        self.loss_D = (self.loss_D_da1 + self.loss_D_da2 + self.loss_D_da3) * 2
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], False)

        # 更新生成器的参数
        self.optimizer.zero_grad()
        self.backward()  # 计算生成器的参数的梯度
        self.optimizer.step()  # 更新参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netpixel_discriminator, self.netfc_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # 计算判别器的梯度
        self.optimizer_D.step()  # update weights

