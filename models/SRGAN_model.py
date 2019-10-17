from .base_model import BaseModel
from . import networks
class SRGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["d", "g"]
        self.visual_names = ["lr_image", "hr_image", "output_img"]
        self.model_names = ['generator', 'discriminator']
        self.netgenerator = networks.Srgan_G(opt.upscale_factor, self.gpu_ids)
        self.netdiscriminator = networks.Srgan_D(self.gpu_ids)
        #self.netresfcn50.to(self.device)
        if self.isTrain:
            self.loss_function = nn.CrossEntropyLoss().to(self.device)
            self.optimizer = torch.optim.Adam(self.netresfcn50.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
