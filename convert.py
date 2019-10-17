from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from PIL import Image as m
import numpy as np
import cv2


if __name__ == '__main__':
    # 加载设置
    opt = TestOptions().parse()

    # 加载数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # 创建模型
    model = create_model(opt)

    # 恢复权重
    model.setup(opt)
    output_dir = "./datasets/mass_transfored/fakeB"
    for i, data in enumerate(dataset):
        image_name_A = data["A_paths"][0].split("/")[-1]
        image_name_B = data["B_paths"][0].split("/")[-1]
        if image_name_A[-5] != image_name_B[-5]:
            print(image_name_A + "is not" + image_name_B)
            break
        model.set_input(data)
        model.forward()
        # result data(image label)
        fake_B = model.fake_B
        imageB_np = util.tensor2im(fake_B)
        label_data = np.squeeze(data["B"].numpy())

        # result_paths(image, label)

        image_out = output_dir + "/images/" + image_name_A
        label_out = output_dir + "/labels/" + image_name_B
        # save images
        cv2.imwrite(image_out, imageB_np)
        cv2.imwrite(label_out, label_data)
        print(i + 1)


