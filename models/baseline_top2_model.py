import torch
import itertools
from .base_model import BaseModel
from . import networks
from torch import nn


class BaselineTop2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--num_classes', type=int, default=2, help='for fcn determining the class number')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["total"]    # "loss_"
        self.visual_names = ["imagelr", "label", "prediction"]  # ""
        if self.isTrain:
            self.visual_names += ["imagesr", "image"]

        self.model_names = ['generator']  # "net"

        self.netgenerator = networks.generator(num_cls=opt.num_classes, gpu_ids=self.gpu_ids)

        if self.isTrain:

            # 语义分割损失
            self.loss_function = nn.CrossEntropyLoss().to(self.device)

            # 超分辨损失函数
            self.generator_criterion = networks.GeneratorLoss().to(self.device)

            # 优化器
            self.optimizer = torch.optim.Adam(self.netgenerator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0005)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        if self.isTrain:
            self.imagelr = input["A"].to(self.device)  # [3, 60, 60] 高分的biculic下采样图
            self.label = input["B"].to(self.device)  # [1, 240, 240] 低分辨率图像对应的上采样label图
            self.image = input["C"].to(self.device)  # [3, 240, 240] 高分图像
        else:
            self.imagelr = input["A"].to(self.device)  # [3, 60, 60] 高分图像
            self.label = input["B"].to(self.device)  # [1, 240, 240] 高分label图

    def forward(self):
        _, self.pre, self.imagesr, _ = self.netgenerator(self.imagelr)
        self.prediction = self.pre.data.max(1)[1].unsqueeze(1)

    def backward(self):
        """计算损失：image_fakeB分割损失"""

        # image_fakeB分割损失
        self.loss_cross_entropy = self.loss_function(self.pre, self.label.long().squeeze(1))
        self.loss_sr = self.generator_criterion(self.imagesr, self.image, is_sr=True)
        self.loss_total = self.loss_cross_entropy + self.loss_sr
        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        # 更新参数
        self.optimizer.zero_grad()
        self.backward()   # 计算参数的梯度
        self.optimizer.step() # 更新参数