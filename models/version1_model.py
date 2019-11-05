import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn
from .networks import Regularization
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
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["resfcn50"]    # "loss_"
        self.visual_names = ["image_A", "label_A", "prediction", "image_B_lr","fake_B_lr",  "image_B_hr"]  # ""
        self.model_names = ['feature_encoder', 'classifier', 'da_decoder', 'da_discriminator']  # "net"

        self.netfeature_encoder = networks.resfcn50(opt.is_restore_from_imagenet, opt.resnet_weight_path, opt.num_classes, self.gpu_ids)
        self.netclassifier = Regularization(model=self.netresfcn50, weight_decay=0.05).to(self.device) # L2正则化
        #self.netresfcn50.to(self.device)
        if self.isTrain:
            self.netda_decoder = ""
            self.netda_discriminator = ""
            self.fake_B_pool = ImagePool(opt.pool_size)

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # discriminator loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            # 数据一致性损失
            self.criterionIdt = torch.nn.L1Loss()

            # 优化器
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netfeature_encoder.parameters(), self.netda_decoder.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netda_discriminator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.Adam(itertools.chain(self.netfeature_encoder.parameters(), self.netclassifier.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer)
    def set_input(self, input):
        self.image_A = input["A"].to(self.device) # [3, 240, 240]
        self.label_A = input["C"].to(self.device) # [1, 240, 240]
        self.image_B_lr = input['B'].to(self.device) # [3, 240, 240]
        self.image_B_hr = input['D'].to(self.device) # [3, 800, 800]
    def forward(self):
        self.internal_feature = self.netfeature_encoder(self.image_A)
        self.fake_B_lr = self.netda_decoder(self.internal_feature)
        self.pre = self.netclassifier(self.internal_feature)
        self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        # forward
        self.forward()
        self.optimizer.zero_grad()

        # calculate gradients
        self.CrossEntropyLoss = self.loss_function(self.pre, self.label.long().squeeze(1))
        self.L2_loss = self.L2_loss_net(self.netresfcn50)
        self.loss_resfcn50 = self.CrossEntropyLoss + self.L2_loss
        self.loss_resfcn50.backward()

        self.optimizer.step()  # update resfcn50'weights
