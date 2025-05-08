from __future__ import print_function, division
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = image.resize((self.output_size, self.output_size), Image.BILINEAR)
        label = label.resize((self.output_size, self.output_size), Image.NEAREST)

        return {'image': image, 'label': label}

class ToTensorLab(object):
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        tmp_img = np.array(image)
        tmp_lbl = np.array(label)

        if len(tmp_img.shape) == 2:
            tmp_img = tmp_img[:, :, np.newaxis]
            tmp_img = np.repeat(tmp_img, 3, axis=2)

        tmp_img = tmp_img / 255.0
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = torch.from_numpy(tmp_img).float()

        if len(tmp_lbl.shape) == 2:
            tmp_lbl = tmp_lbl[:, :, np.newaxis]

        tmp_lbl = tmp_lbl.transpose((2, 0, 1))
        tmp_lbl = torch.from_numpy(tmp_lbl).float()

        return {'image': tmp_img, 'label': tmp_lbl}
