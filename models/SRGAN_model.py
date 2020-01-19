from .base_model import BaseModel
from . import networks_now
import torch

class SRGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["d", "g"]
        self.visual_names = ["lr_image", "hr_image", "sr_img"]
        self.model_names = ['generator', 'discriminator']
        self.netgenerator = networks_now.Srgan_Generator(opt.upscale_factor, self.gpu_ids)
        self.netdiscriminator = networks_now.Srgan_Discriminator(self.gpu_ids)
        if self.isTrain:
            #self.g_loss = networks.Srgan_Gloss(self.gpu_ids)
            self.g_loss = networks_now.GeneratorLoss().to(self.device)
            self.optimizerG = torch.optim.Adam(self.netgenerator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizerD = torch.optim.Adam(self.netdiscriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizerG)
            self.optimizers.append(self.optimizerD)
    def set_input(self, input):
        self.lr_image = input["A"].to(self.device) # [0,1]
        self.hr_image = input["B"].to(self.device) # [0,1]
        self.image_paths = input['A_paths']
    def forward(self):
        self.sr_img = self.netgenerator(self.lr_image) # [0,1]
        if self.isTrain:
            self.real_out = self.netdiscriminator(self.hr_image).mean() # scalar
            self.fake_out = self.netdiscriminator(self.sr_img).mean() # scalar

    def optimize_parameters(self):
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        # forward
        self.forward()
        self.netdiscriminator.zero_grad()
        self.loss_d = 1 - self.real_out + self.fake_out
        self.loss_d.backward(retain_graph=True)
        self.optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        self.netgenerator.zero_grad()
        self.loss_g = self.g_loss(self.fake_out, self.sr_img, self.hr_image)
        self.loss_g.backward()
        self.optimizerG.step()

