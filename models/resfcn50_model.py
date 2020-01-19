import torch
import itertools
from .base_model import BaseModel
from . import networks_now
from torch import nn
from .networks_now import Regularization


class Resfcn50Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for fcn determining the class number')
        parser.add_argument('--is_restore_from_imagenet', type=bool, default=True,
                            help='for fcn determining whether or not to restore resnet50 from imagenet')
        parser.add_argument('--resnet_weight_path', type=str, default='./resnetweight/',
                            help='the path to renet_weight from imagenet')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["resfcn50"]
        self.visual_names = ["image", "label", "prediction"]
        self.model_names = ['resfcn50']
        self.netresfcn50 = networks_now.resfcn50(opt.is_restore_from_imagenet, opt.resnet_weight_path, opt.num_classes, self.gpu_ids)
        self.L2_loss_net = Regularization(model=self.netresfcn50, weight_decay=0.05).to(self.device) # L2正则化
        #self.netresfcn50.to(self.device)
        if self.isTrain:
            self.loss_function = nn.CrossEntropyLoss().to(self.device)
            self.optimizer = torch.optim.Adam(self.netresfcn50.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
    def set_input(self, input):
        self.image = input["A"].to(self.device)
        self.label = input["B"].to(self.device)
        self.image_paths = input['A_paths']
    def forward(self):
        self.pre = self.netresfcn50(self.image)
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
