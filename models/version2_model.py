import torch
import itertools
from .base_model import BaseModel
from . import networks_now
from torch import nn
from util.image_pool import ImagePool

class Version2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for fcn determining the class number')
        parser.add_argument('--is_restore_from_imagenet', type=bool, default=True,
                            help='for fcn determining whether or not to restore resnet50 from imagenet')
        parser.add_argument('--resnet_weight_path', type=str, default='./resnetweight/',
                            help='the path to renet_weight from imagenet')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        if is_train:
            parser.add_argument('--lambda_class', type=float, default=1.0, help='weight for the segmentation loss')
            parser.add_argument('--lambda_featureDA', type=float, default=0.001,
                                help='weight for image_A faking the feature discriminator')
            parser.add_argument('--lambda_iamgeDA', type=float, default=0.001,
                                help='weight for image_A faking the image discriminator')
            parser.add_argument('--lambda_identityA', type=float, default=1.0,
                                help='weight for image_A identity loss')
            parser.add_argument('--lambda_identityB', type=float, default=1.0,
                                help='weight for image_B identity loss')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["total_loss", "da_D"]    # "loss_"
        self.visual_names = ["image_A", "label_A", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["domain_hotmap_A", "image_fakeB", "image_B"]
        self.model_names = ['backbone', 'psp_classifier']  # "net"

        if self.isTrain:
            self.model_names += ['aspp_discriminator', 'sr_decoder', 'sr_discriminator']

        self.netbackbone = networks_now.backbone_resnet50fcn(pretrained_backbone=opt.is_restore_from_imagenet,
                                                             resnet_weight_path=opt.resnet_weight_path,
                                                             gpu_ids=self.gpu_ids)
        self.netpsp_classifier = networks_now.psp_classifier(opt.num_classes, self.gpu_ids)

        if self.isTrain:
            self.netaspp_discriminator = networks_now.aspp_discriminator(gpu_ids=self.gpu_ids)

            # 超分辨
            self.netsr_decoder = networks_now.sr_decoder(gpu_ids=self.gpu_ids)
            self.netsr_discriminator = networks_now.define_D(input_nc=3, ndf=64, netD="basic", gpu_ids=self.gpu_ids)


            self.image_B_pool = ImagePool(opt.pool_size)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 判别器损失
            self.criterionGAN = networks_now.GANLoss(opt.gan_mode).to(self.device)

            # 超分辨一直性损失
            self.sr_loss = nn.MSELoss().to(self.device)


            # 优化器
            self.optimizer = torch.optim.Adam(itertools.chain(self.netbackbone.parameters(), self.netpsp_classifier.parameters(), self.netsr_decoder.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netaspp_discriminator.parameters(), self.netsr_discriminator.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.image_A = input["A"].to(self.device) # [3, 240, 240]
        self.label_A = input["B"].to(self.device) # [1, 240, 240]
        if self.isTrain:
            self.image_B = input['C'].to(self.device) # [3, 240, 240]
    def forward(self):
        if self.isTrain:
            ##########################
            # 把image_A通过：所有网络(除了PSP分类器网络)
            ##########################

            # iamge_A 通过：特征提取器网络
            self.feature_A = self.netbackbone(self.image_A)

            # iamge_A 通过分类器
            #self.pre = self.netpsp_classifier(self.feature_A)
            #self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

            # image_A 通过：ASPP特征域判别器网络
            self.domain_hotmap_A = self.netaspp_discriminator(self.feature_A) #[0, 1]

            # image_A 通过：超分辨解码器网络
            self.image_fakeB = self.netsr_decoder(self.feature_A).detach() #(-1, 1)

            # image_A 通过：超分辨判别器网络
            self.outD_image_A = self.netsr_discriminator(self.image_fakeB)

            ##########################
            # 把image_fakeB通过：特征提取器网络、PSP分类器网络、ASPP特征域判别器网络
            ##########################

            # image_fakeB通过：特征提取器网络
            self.feature_fakeB = self.netbackbone(self.image_fakeB)

            # image_fakeB通过：PSP分类器网络
            self.pre = self.netpsp_classifier(self.feature_fakeB)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

            # iamge_fakeB通过：ASPP特征域判别器网络
            #self.outD_image_fakeB = self.netaspp_discriminator(self.feature_fakeB) # [0, 1]


            ##########################
            # 把image_B通过：特征提取器网络，超分辨解码器网络；超分辨判别器网络
            ##########################

            # 首先从缓存池中取出image_B
            self.image_B = self.image_B_pool.query(self.image_B)

            # iamge_B通过：特征提取器网络
            self.feature_B = self.netbackbone(self.image_B)
            self.outD_image_fakeB = self.netaspp_discriminator(self.feature_B)  # [0, 1]
            # image_B通过： 超分辨解码器网络
            self.id_image_B = self.netsr_decoder(self.feature_B)  # (-1, 1)

            # image_B通过: 超分辨判别器网络
            self.outD_image_B = self.netsr_discriminator(self.image_B)

        else:
            self.feature_A = self.netbackbone(self.image_A)
            self.pre = self.netpsp_classifier(self.feature_A)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算五个损失：image_A欺骗特征域判别器损失、iamge_A 欺骗超分辨判别器损失
                         iamge_A图像图像一致性损失、image_B图像一致性损失、
                         image_fakeB分割损失"""

        lambda_featureDA = self.opt.lambda_featureDA
        lambda_iamgeDA = self.opt.lambda_iamgeDA
        lambda_identityA = self.opt.lambda_identityA
        lambda_identityB = self.opt.lambda_identityB
        lambda_class = self.opt.lambda_class

        # image_A欺骗特征域判别器损失
        self.loss_featureDA = self.criterionGAN(self.domain_hotmap_A, True) * lambda_featureDA

        # iamge_A 欺骗超分辨判别器损失
        self.loss_iamgeDA = self.criterionGAN(self.outD_image_A, True) * lambda_iamgeDA

        #iamge_A图像图像一致性损失
        self.loss_identityA = self.sr_loss(self.image_fakeB, self.image_A) * lambda_identityA

        # image_B图像一致性损失
        self.loss_identityB = self.sr_loss(self.id_image_B, self.image_B) * lambda_identityB

        # image_fakeB分割损失
        self.CrossEntropyLoss = self.loss_function(self.pre, self.label_A.long().squeeze(1)) * lambda_class

        # 求所有的损失和
        self.loss_total_loss = self.loss_featureDA + self.loss_iamgeDA + self.loss_identityA + self.loss_identityB + self.CrossEntropyLoss

        self.loss_total_loss.backward(retain_graph=True) #retain_graph=True
    def backward_D(self):
        """计算D的损失"""
        # 特征域转换的判别器
        self.loss_D_real = self.criterionGAN(self.outD_image_fakeB, True)
        self.loss_D_fake = self.criterionGAN(self.domain_hotmap_A, False)

        # 超分判别器
        self.loss_D_fake_sr = self.criterionGAN(self.outD_image_A, False)
        self.loss_D_real_sr = self.criterionGAN(self.outD_image_B, True)

        self.loss_da_D = (self.loss_D_real + self.loss_D_fake + self.loss_D_fake_sr + self.loss_D_real_sr) * 0.25
        self.loss_da_D.backward()

    def optimize_parameters(self):

        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netaspp_discriminator, self.netsr_discriminator], False)

        # 更新（特征提取器，分类器, 超分解码器）的参数
        self.optimizer.zero_grad()
        self.backward()   # 计算 （特征提取器，分类器， 超分解码器）的参数的梯度
        self.optimizer.step() # 更新 （特征提取器，分类器， 超分解码器） 的参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netaspp_discriminator, self.netsr_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()      # 计算判别器的梯度
        self.optimizer_D.step()  # update D_A and D_B's weights
