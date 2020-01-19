import torch
import itertools
from .base_model import BaseModel
from . import networks_now
from torch import nn
from util.image_pool import ImagePool

class Version4Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for determining the class number')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        if is_train:
            parser.add_argument('--lambda_class', type=float, default=1.0, help='weight for the segmentation loss')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["total_loss", "da_D"]    # "loss_"
        self.visual_names = ["imageB", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imageA", "fakeB", "recA", "fakeA", "recB", "imagelrB", "imagesrB", "reclrB"]

        self.model_names = ['G2', 'psp_classifier']  # "net"

        if self.isTrain:
            self.model_names += ['G1', 'G1_discriminator', 'G2_discriminator']

        self.netG2 = networks_now.v4_G2net(gpu_ids=self.gpu_ids)
        self.netpsp_classifier = networks_now.psp_classifier(opt.val_size, opt.num_classes, self.gpu_ids)

        if self.isTrain:
            self.netG1 = networks_now.v4_G1net(gpu_ids=self.gpu_ids)

            self.netG1_discriminator = networks_now.define_D(input_nc=3, ndf=64, netD="basic", gpu_ids=self.gpu_ids)
            self.netG2_discriminator = networks_now.define_D(input_nc=3, ndf=64, netD="basic", gpu_ids=self.gpu_ids)
            self.image_B_pool = ImagePool(opt.pool_size)
            self.image_A_pool = ImagePool(opt.pool_size)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 判别器损失
            self.criterionGAN = networks_now.GANLoss(opt.gan_mode).to(self.device)

            # 超分辨损失
            self.sr_loss = nn.MSELoss().to(self.device)

            self.Cycle_loss = torch.nn.L1Loss()



            # 优化器
            self.optimizer = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters(), self.netpsp_classifier.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netG1_discriminator.parameters(), self.netG2_discriminator.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            self.imageA = input["A"].to(self.device)  # [3, 240, 240] 低分辨率图像
            self.label = input["B"].to(self.device)  # [1, 800, 800] 低分辨率图像对应的上采样label图
            self.imagelrB = input["C"].to(self.device)  # [3, 240, 240] 高分的biculic下采样图
            self.imageB = input["D"].to(self.device)  # [3, 800, 800] 高分图像
        else:
            self.imageB = input["A"].to(self.device)  # [3, 800, 800] 高分图像
            self.label = input["B"].to(self.device)  # [1, 800, 800] 高分label图


    def forward(self):
        if self.isTrain:
            ##########################
            # imageA cycle: imageA[240, 240]=>fakeB[800, 800]=>recA[240, 240]
            ##########################

            # iamge_A 通过G1
            self.fakeB = self.netG1(self.imageA, is_B=False)

            # fakeB 通过G2
            self.fakeB_feature, self.recA = self.netG2(self.fakeB, is_B=False)

            # fakeB_feature 通过分类器
            self.pre = self.netpsp_classifier(self.fakeB_feature)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)


            ##########################
            # image_B cycle: imageB[800, 800]=>fakeA[240, 240]=>recB[800, 800]
            ##########################
            # fakeB 通过G2
            _, self.fakeA = self.netG2(self.imageB, is_B=False)
            # iamge_A 通过G1
            self.recB = self.netG1(self.fakeA, is_B=False)


            ##########################
            # sr cycle:  imagelrB[240, 240]=>imagesrB[800, 800]=>reclrB[240, 240]
            ##########################

            self.imagesrB = self.netG1(self.imagelrB, is_B=True)
            _, self.reclrB = self.netG2(self.imagesrB, is_B=True)

            ##########################
            # fakeB, imageB_new 通过G1_discriminator
            ##########################
            # 首先从缓存中拿出imageB_new
            self.imageB_new = self.image_B_pool.query(self.imageB)
            self.outD_G1_true = self.netG1_discriminator(self.imageB_new)
            self.outD_G1_false = self.netG1_discriminator(self.fakeB)

            ###########################
            # fakeA, imageA_new 通过G2_discriminator
            ###########################
            self.imageA_new = self.image_A_pool.query(self.imageA)
            self.outD_G2_true = self.netG2_discriminator(self.imageA_new)
            self.outD_G2_false = self.netG2_discriminator(self.fakeA)

        else:
            self.imageB_feature, _ = self.netG2(self.imageB)
            self.pre = self.netpsp_classifier(self.imageB_feature)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算六个损失"""


        # GAN loss D_G1(G1(A))
        self.loss_G_A = self.criterionGAN(self.outD_G1_false, True)

        # GAN loss D_G2(G2(B))
        self.loss_G_B = self.criterionGAN(self.outD_G2_false, True)


        # A cycle loss
        self.loss_cycle_A = self.Cycle_loss(self.recA, self.imageA)

        # B cycle loss
        self.loss_cycle_B = self.Cycle_loss(self.recB, self.imageB)

        # sr loss
        self.loss_sr = self.sr_loss(self.imagesrB, self.imageB)

        # sr identity loss
        self.loss_sr_id = self.Cycle_loss(self.reclrB, self.imagelrB)

        # classifier loss
        self.CrossEntropyLoss = self.loss_function(self.pre, self.label.long().squeeze(1))

        # 求所有的损失和
        self.loss_total_loss = self.loss_sr * 5 + self.loss_sr_id * 5 + self.loss_G_A +  self.loss_G_B + self.loss_cycle_A  * 10 + self.loss_cycle_B * 10  #+ self.CrossEntropyLoss * 5

        self.loss_total_loss.backward(retain_graph=True) #retain_graph=True
    def backward_D_A(self):
        """计算D的损失"""
        # 特征域转换的判别器
        self.loss_D_G1_real = self.criterionGAN(self.outD_G1_true, True)
        self.loss_D_G1_fake = self.criterionGAN(self.outD_G1_false, False)

        self.loss_da_D = (self.loss_D_G1_real + self.loss_D_G1_fake) * 0.5
        self.loss_da_D.backward()
    def backward_D_B(self):
        """计算D的损失"""
        # 超分判别器
        self.loss_D_G2_fake_sr = self.criterionGAN(self.outD_G2_false, False)
        self.loss_D_G2_real_sr = self.criterionGAN(self.outD_G2_true, True)

        self.loss_da_D = (self.loss_D_G2_fake_sr + self.loss_D_G2_real_sr) * 0.5
        self.loss_da_D.backward()

    def optimize_parameters(self):

        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netG1_discriminator, self.netG2_discriminator], False)

        # 更新（G1, G2）的参数
        self.optimizer.zero_grad()
        self.backward()   # 计算 （特征提取器，分类器， 超分解码器）的参数的梯度
        self.optimizer.step() # 更新 （特征提取器，分类器， 超分解码器） 的参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netG1_discriminator, self.netG2_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()      # 计算判别器的梯度
        self.backward_D_B()
        self.optimizer_D.step()  # update D_A and D_B's weights
