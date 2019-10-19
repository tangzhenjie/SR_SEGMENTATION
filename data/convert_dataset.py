from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image as m
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
def transform(image, mask, opt):
    if not opt.no_crop:
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(opt.A_crop_size, opt.A_crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)



    # 插值
    #if opt.inter_method_image != "":
    #    if opt.inter_method_image == "bilinear":
    #        interfunction_image = transforms.Resize(opt.B_crop_size)
    #        image = interfunction_image(image)
    if opt.inter_method_label != "":
        if opt.inter_method_label == "nearest":
            mask = mask.resize((opt.B_crop_size, opt.B_crop_size))
    mask = np.array(mask).astype(np.long)
    #nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Transform to tensor
    image = TF.to_tensor(image)
    #image = nomal_fun_image(image)
    mask = TF.to_tensor(mask)
    return image, mask

class ConvertDataset(BaseDataset):
    """load train and val for segmentation network or for convert trainA to B using trained cycle_gan
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--A_crop_size', type=int, default=60, help='A crop to this size')
        parser.add_argument('--B_crop_size', type=int, default=240, help='B crop to this size')
        parser.add_argument('--no_crop', type=bool, default=False,
                            help='crop the A and B according to the special datasets params  [crop | none],')
        parser.add_argument('--inter_method_image', type=str, default='bilinear', help='the image Interpolation method')
        parser.add_argument('--inter_method_label', type=str, default='nearest', help='the label Interpolation method')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc  for the directory name')
        return parser
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot + "/" + opt.phase + 'A/images'  # create a path '/trainA/images/*.png'
        self.dir_B = opt.dataroot + "/" + opt.phase + 'A/labels'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size)) # load images from '/trainA/images/*.tif'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.A_size]
        A_img = m.open(A_path).convert('RGB')
        B_img = m.open(B_path).convert('L')

        A, B = transform(A_img, B_img, self.opt)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size