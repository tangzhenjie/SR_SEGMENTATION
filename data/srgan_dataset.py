from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor()
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
class SrganDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--B_crop_size', type=int, default=240, help='crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc  for the directory name')
        return parser
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_B = opt.dataroot + "/" + opt.phase + 'B/images'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if self.opt.phase == "train":
            self.hr_transform = train_hr_transform(opt.B_crop_size)
            self.lr_transform = train_lr_transform(opt.B_crop_size, opt.upscale_factor)
    def __getitem__(self, index):
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        if self.opt.phase == "train":
            hr_image = self.hr_transform(Image.open(B_path).convert('RGB'))
            lr_image = self.lr_transform(hr_image)

            return {'A': lr_image, 'B': hr_image, 'A_paths': B_path, 'B_paths': B_path}
        else:
            lr_scale = Resize(self.opt.B_crop_size // self.opt.upscale_factor, interpolation=Image.BICUBIC)
            hr_scale = Resize(self.opt.B_crop_size, interpolation=Image.BICUBIC)
            hr_image = RandomCrop(self.opt.B_crop_size)(Image.open(B_path).convert('RGB'))
            lr_image = lr_scale(hr_image)
            hr_restore_img = hr_scale(lr_image)
            return {'A': ToTensor()(lr_image), 'B': ToTensor()(hr_image), 'C':ToTensor()(hr_restore_img), 'A_paths': B_path, 'B_paths': B_path}


    def __len__(self):
        return self.B_size
