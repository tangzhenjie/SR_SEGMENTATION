from .base_model import BaseModel
from . import networks_now
from util.image_pool import ImagePool
from . import networks_now
import torch
import itertools


class SISRModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # cycle gan
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')

        # segmentation net param
        parser.add_argument('--num_classes', type=int, default=2, help='for fcn determining the class number')
        parser.add_argument('--is_restore_from_imagenet', type=bool, default=True,
                            help='for fcn determining whether or not to restore resnet50 from imagenet')
        parser.add_argument('--resnet_weight_path', type=str, default='./resnetweight/',
                            help='the path to renet_weight from imagenet')

        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # 设置要显示的loss
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'Seg']

        # 设置要显示的图像结果
        visual_names_A = ['real_A', 'fake_B', 'rec_A', ]
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        # 添加上标签和预测结果
        if self.isTrain:
            self.visual_names = visual_names_A + visual_names_B + ['label', 'prediction']
        else:
            self.visual_names = ['real_A', 'label', 'prediction']

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'Seg']
        else:  # during test time, only load Gs
            self.model_names = ['Seg']

        # 构建网络
        if self.isTrain:
            self.netG_A = networks_now.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks_now.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD_A = networks_now.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks_now.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # 定义分割网络
        self.netSeg = networks_now.resfcn50(opt.is_restore_from_imagenet, opt.resnet_weight_path, opt.num_classes, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks_now.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.CrossEntropyLoss()

            # 正则化loss
            self.L2_loss_net = networks_now.Regularization(model=self.netSeg, weight_decay=0.05)  # L2正则化
            self.L2_loss_net.to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netSeg.parameters(), lr=opt.lr_seg, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        if self.isTrain:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.label = input["C"].to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.label = input["B"].to(self.device)
            self.image_paths = input['A_paths']
    def forward(self):
        if self.isTrain:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
            self.pre = self.netSeg(self.fake_B)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)  # for visual
        else:
            self.pre = self.netSeg(self.real_A)
            self.prediction = self.pre.data.max(1)[1].unsqueeze(1)  # for visual

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

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B


        # 计算分割损失，来使G和Seg网络，使分割正确
        self.CrossEntropyLoss = self.criterionSeg(self.pre, self.label.long().squeeze(1))
        self.L2_loss = self.L2_loss_net(self.netSeg)
        self.loss_Seg = self.CrossEntropyLoss + self.L2_loss

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_Seg
        self.loss_G.backward()

    def backward_S(self):
        """Calculate the loss for generators netSeg"""
        # compute the loss
        self.CrossEntropyLoss = self.criterionSeg(self.pre, self.label.long().squeeze(1))
        self.L2_loss = self.L2_loss_net(self.netSeg)
        self.loss_Seg = self.CrossEntropyLoss + self.L2_loss
        self.loss_Seg.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # backward G & Seg
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_S.zero_grad()

        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_S.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights




