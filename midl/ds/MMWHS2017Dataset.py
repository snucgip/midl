"""
A pytorch dataset for Multi-Modality Whole Heart Segmentation 2017

Read raw format of cardiac CT, MR data and convert to numpy array

Author      : Sanguk Park
Version     : 0.1
"""

import torch
from torch.utils.data import Dataset
import scipy.ndimage as ni
from skimage.transform import resize
import numpy as np

import glob
import os

from midl.utils import image_processing as iu
from midl.utils import matrix_processing as mu


class MMWHS2017Dataset(Dataset):
    def __init__(self,
                 width: int,
                 height: int,
                 depth: int,
                 path_image_dir,
                 path_label_dir=None,
                 aug=True):
        """

        :param width: target width of image matrix
        :param height: target height of image matrix
        :param depth: target depth of image matrix
        :param path_image_dir:
        :param path_label_dir:
        :param aug:

        """

        self.path_images = glob.glob(path_image_dir+'/*')
        self.path_labels = None

        if path_label_dir is not None:
            self.path_labels = glob.glob(path_label_dir+'/*')

            assert len(self.path_images) == len(self.path_labels)

        self.n_data = len(self.path_images)

        self.shape = (depth, height, width)
        self.aug = aug

    def __len__(self):
        return self.n_data
    
    def __getitem__(self, item):
        """
        Resize the image to fixed size (D, H, W) format.

        :param item:
        :return:
        """
        sample = {}

        image = iu.load_raw_image(self.path_images[item])
        image = resize(image, self.shape)
        image = iu.normalize(image, -300, 1000)

        sample['image'] = image
        sample['filename'] = os.path.basename(self.path_images[item])

        if self.path_labels is not None:
            label = iu.load_raw_image(self.path_labels[item])
            label = resize(label, self.shape, order=0, preserve_range=True, anti_aliasing=False)
            label = label.astype(np.int16)

            values = np.unique(label)

            weights = []
            for i in range(len(values)):
                weights.append(len(label[label == values[i]]))
                if i != 0:
                    label[label == values[i]] = i
            max_weight = max(weights)
            weights = np.array([max_weight/x for x in weights])

            sample['label'] = label
            sample['weights'] = weights

        return sample


if __name__ == "__main__":
    ds = MMWHS2017Dataset(128, 128, 64,
                          'D:\data\Cardiac\MMWHS2017\ct_train_raw\image',
                          'D:\data\Cardiac\MMWHS2017\ct_train_raw\label')

    for sample in ds:
        print(sample['image'].shape)
        break
