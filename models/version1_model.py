import torch
import itertools
from .base_model import BaseModel
from . import networks_now
from torch import nn
from .networks_now import Regularization
from util.image_pool import ImagePool


class Version1Model(BaseModel):
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
            parser.add_argument('--lambda_class', type=float, default=10.0, help='weight for the segmentation loss')
            parser.add_argument('--lambda_da', type=float, default=5.0, help='weight for Domain Adaptation loss ')
            parser.add_argument('--lambda_identity', type=float, default=5.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_fixpoint', type=float, default=5.0, help='weight for the fixpoint_loss')
            parser.add_argument('--lambda_sisr', type=float, default=1.0, help='weight for super resolution loss')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["total_loss", "da_D"]    # "loss_"
        self.visual_names = ["image_A", "label_A", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["fake_B_lr",  "image_B_lr", "image_B_hr"]
        self.model_names = ['domainB_encoder', 'share_feature', 'classifier']  # "net"

        if self.isTrain:
            self.model_names += ['domainA_encoder', 'da_decoder', 'da_discriminator']

        #self.netfeature_encoder = networks.feature_encoder(opt.is_restore_from_imagenet, opt.resnet_weight_path, self.gpu_ids)
        #self.netfeature_encoder = networks.feature_encoder(gpu_ids=self.gpu_ids)
        self.netdomainB_encoder = networks_now.domainB_encoder(gpu_ids=self.gpu_ids)
        self.netshare_feature = networks_now.share_feature(gpu_ids=self.gpu_ids)

        self.netclassifier = networks_now.classifier_segment(opt.num_classes, self.gpu_ids)
        if self.isTrain:
            self.netdomainA_encoder = networks_now.domainA_encoder(gpu_ids=self.gpu_ids)
            self.netda_decoder = networks_now.da_decoder(self.gpu_ids)
            self.netda_discriminator = networks_now.define_D(input_nc=3, ndf=64, netD='n_layers', gpu_ids=self.gpu_ids)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 判别器损失
            self.criterionGAN = networks_now.GANLoss(opt.gan_mode).to(self.device)

            # 数据一致性损失
            self.criterionIdt = torch.nn.L1Loss()

            # fixpoint损失
            self.fixpointIdt = torch.nn.MSELoss()

            # 优化器
            self.optimizer = torch.optim.Adam(itertools.chain(self.netdomainA_encoder.parameters(), self.netdomainB_encoder.parameters(),  self.netshare_feature.parameters(),  self.netda_decoder.parameters(), self.netclassifier.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netda_discriminator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer = torch.optim.Adam(itertools.chain(self.netfeature_encoder.parameters(), self.netclassifier.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer)
    def set_input(self, input):
        self.image_A = input["A"].to(self.device) # [3, 240, 240]
        self.label_A = input["B"].to(self.device) # [1, 240, 240]
        if self.isTrain:
            self.image_B_lr = input['C'].to(self.device) # [3, 240, 240]
            self.image_B_hr = input['D'].to(self.device) # [3, 800, 800]
    def forward(self):
        if self.isTrain:
            self.feature_A = self.netdomainA_encoder(self.image_A)
            self.feature_A_share = self.netshare_feature(self.feature_A)
            self.pre = self.netclassifier(self.feature_A_share)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

            self.feature_B = self.netdomainB_encoder(self.image_B_lr)
            self.feature_B_share = self.netshare_feature(self.feature_B)
            self.fake_B_lr = self.netda_decoder(self.feature_A_share)
            self.true_B_lr = self.netda_decoder(self.feature_B_share)
        else:
            self.feature_B = self.netdomainB_encoder(self.image_A)
            self.feature_B_share = self.netshare_feature(self.feature_B)
            self.pre = self.netclassifier(self.feature_B_share)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算三个损失：分割损失，域转换损失，图像一致性损失"""

        lambda_class = self.opt.lambda_class
        lambda_da = self.opt.lambda_da
        lambda_identity = self.opt.lambda_identity
        lambda_fixpoint = self.opt.lambda_fixpoint

        # 图像一致性损失
        if lambda_identity > 0:
            # 提取image_B_lr特征
            self.loss_idt_B = self.criterionIdt(self.true_B_lr, self.image_B_lr) * lambda_identity
        else:
            self.loss_idt_B = 0

        # 计算 fixpoint 损失
        self.feature_fake_B_lr = self.netdomainB_encoder(self.fake_B_lr)
        self.feature_fake_B_lr_share = self.netshare_feature(self.feature_fake_B_lr)
        self.loss_fixpoint_A = self.fixpointIdt(self.feature_fake_B_lr_share, self.feature_A_share) * lambda_fixpoint
        self.feature_true_B_lr = self.netdomainB_encoder(self.true_B_lr)
        self.feature_true_B_lr_share = self.netshare_feature(self.feature_true_B_lr)
        self.loss_fixpoint_B = self.fixpointIdt(self.feature_true_B_lr_share, self.feature_B_share) * lambda_fixpoint

        self.loss_fixpoint = self.loss_fixpoint_A + self.loss_fixpoint_B

        # 域转换损失
        self.loss_da = self.criterionGAN(self.netda_discriminator(self.fake_B_lr), True) * lambda_da

        # 分割损失
        self.CrossEntropyLoss = self.loss_function(self.pre, self.label_A.long().squeeze(1)) * lambda_class

        self.loss_total_loss = self.loss_idt_B + self.loss_da + self.CrossEntropyLoss + self.loss_fixpoint
        self.loss_total_loss.backward()
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def backward_D(self):
        """计算D的损失"""
        fake_B = self.fake_B_pool.query(self.fake_B_lr)
        self.loss_da_D = self.backward_D_basic(self.netda_discriminator, self.image_B_lr, fake_B)


    def optimize_parameters(self):

        self.forward()

        # 不求判别器的梯度
        self.set_requires_grad([self.netda_discriminator], False)

        # 更新（特征提取器，分类器， da 解码器）的参数
        self.optimizer.zero_grad()
        self.backward()   # 计算 （特征提取器，分类器， da 解码器）的参数的梯度
        self.optimizer.step() # 更新 （特征提取器，分类器， da 解码器） 的参数

        # 可以求判别器的梯度
        self.set_requires_grad([self.netda_discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()      # 计算判别器的梯度
        self.optimizer_D.step()  # update D_A and D_B's weights
