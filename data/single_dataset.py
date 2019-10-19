from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image as m
import torch
import numpy as np
def get_transformA(opt, convert=True):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
def get_transformB(opt, convert=True):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)
class SingleDataset(BaseDataset):
    """load train and val for segmentation network
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc  for the directory name')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--is_B', type=bool, default=True,
                            help='when opt.phase==train it determines whether or not to use trainB')
        parser.add_argument('--is_fakeB', type=bool, default=False,
                            help='when opt.phase==train it determines whether or not to use the transfored fakeB')
        return parser
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if opt.phase == "train":
            if opt.is_B:
                self.dir_A = opt.dataroot + "/" + opt.phase + 'B/images'  # create a path '/trainA/images/*.png'
                self.dir_B = opt.dataroot + "/" + opt.phase + 'B/labels'  # labels path
            elif opt.is_fakeB:
                self.dir_A = opt.dataroot + "/" + 'fakeB/images'  # create a path '/trainA/images/*.png'
                self.dir_B = opt.dataroot + "/" + 'fakeB/labels'
            else:
                self.dir_A = opt.dataroot + "/" + opt.phase + 'A/images'  # create a path '/trainA/images/*.png'
                self.dir_B = opt.dataroot + "/" + opt.phase + 'A/labels'
        elif opt.phase == "val":
            self.dir_A = opt.dataroot + "/" + opt.phase + 'B/images'  # create a path '/trainA/images/*.png'
            self.dir_B = opt.dataroot + "/" + opt.phase + 'B/labels'

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/images/*.png'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/labels/*.png'
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transformA = get_transformA(opt)
        self.transformB = get_transformB(opt)


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
        # read the image and corresponding to the label
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = m.open(A_path).convert('RGB')
        B_label = np.array(m.open(B_path).convert('L')).astype(np.long)
        # apply image transformation
        A = self.transformA(A_img)
        B = self.transformB(B_label)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)