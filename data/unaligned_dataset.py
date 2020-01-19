from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
from PIL import Image as m
import torchvision.transforms as transforms


def get_transformA(opt, convert=True):
    transform_list = []

    #if not opt.no_crop:
    #    transform_list.append(transforms.RandomCrop(opt.A_crop_size))

    if not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    # Interpolation A to B size
    #if opt.inter_method != "":
    #    if opt.inter_method == "bilinear":
    #        transform_list += [transforms.Resize(opt.B_crop_size)]
    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transformB(opt, convert=True):
    transform_list = []

    #if not opt.no_crop:
    #    transform_list.append(transforms.RandomCrop(opt.B_crop_size))

    #transform_list += [transforms.Resize(opt.A_crop_size, interpolation=m.BICUBIC)]

    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets

    It requires two directories to host training images from domain A '/path/A/train/images
    and from domain B '/path/B/train/images
    You can train the model with flag '--dataroot /path/'
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--A_crop_size', type=int, default=60, help='crop to this size')
        parser.add_argument('--B_crop_size', type=int, default=240, help='crop to this size')
        parser.add_argument('--inter_method', type=str, default='bilinear', help='the Interpolation method')
        parser.add_argument('--no_crop',  type=bool, default=False,
                            help='crop the A and B according to the special datasets params  [crop | none],')
        parser.add_argument('--no_flip', type=bool, default=False,
                            help='if specified, do not flip the images for data augmentation')
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
        self.dir_A = opt.dataroot + "/" + opt.phase + 'A/images'  # create a path '/trainA/images/*.tif'
        self.dir_B = opt.dataroot + "/compress_" + opt.phase + 'B/images'  # create a path '/trainB/images/*.tif'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/trainA/images/*.tif'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/trainA/images/*.tif'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = get_transformA(self.opt)
        self.transform_B = get_transformB(self.opt)

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
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = m.open(A_path).convert('RGB')
        B_img = m.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)