"""
File:		class LiverDataset3D
Language:	Python 3.6.5
Library:	PyTorch 0.4.0
Author:		Minyoung Chung
Date:		2018-05-03
Version:	1.0
Mail:		chungmy.freddie@gmail.com

Copyright (c) 2018 All rights reserved by Bokkmani Studio corp.
"""
import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import Dataset
import glob

import numpy as np
from midl.utils import image_processing as iu
from midl.utils.image_processing import load_raw_image
from midl.utils import matrix_processing as mu
import scipy.ndimage as ni

import os

class LiverRawDataset(Dataset):
    def __init__(self,
                 width,
                 height,
                 depth,
                 path_image_dir,
                 path_label_dir,
                 aug=True,
                 transform=None):
        """
        :param width:                   width of 3d raw data.
        :param height:                  height of 3d raw data.
        :param depth:                   depth of 3d raw data.
        :param root_dir_image:          Root directory path for images.
        :param root_dir_label:          Root directory path for labels.
        :param transform:              Boolean for transformation.
        """

        self.shape = (depth, height, width)
        self.path_image_dir = path_image_dir
        self.path_label_dir = path_label_dir
        self.transform = transform
        self.aug = aug

        # configurations of data set.
        self.paths_image = glob.glob(self.path_image_dir + '/*')
        if path_label_dir:
            self.paths_label = glob.glob(self.path_label_dir + '/*')

        self.num_data = len(self.paths_image)

        # Create image and label stack.
        self.image_stack = np.empty((0, depth, height, width))
        self.label_stack = np.empty((0, depth, height, width))

        self.org_img_size = []
        self.filename = []


        for i in range(len(self.paths_image)):
            # Save filename
            self.filename.append(os.path.basename(self.paths_image[i]))

            # Load a image and a label
            image = load_raw_image(self.paths_image[i])

            image_shape = image.shape
            self.org_img_size.append(image_shape)
            print(image.shape)

            image = iu.resize_image(image, False, [depth, height, width])

            # Save to stacks
            self.image_stack = np.vstack((self.image_stack, np.expand_dims(image, axis=0)))

            if path_label_dir:
                label = load_raw_image(self.paths_label[i])
                label = label.astype(np.uint8)
                label = iu.resize_image(label, True, [depth, height, width], binary_threshold=0.5)
                label = np.clip(label, 0, 1)

                # Save to stacks
                self.label_stack = np.vstack((self.label_stack, np.expand_dims(label, axis=0)))

            print(i)

        print("Done Loaded.")

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):

        # image = read_raw(self.image_paths[item], 'int16', [self.depth, self.height, self.width], image_size)
        # # image = read_raw(self.image_paths[item], 'float32', [self.width, self.height, self.depth], image_size)
        # label = read_raw(self.label_paths[item], 'uint8', [self.depth, self.height, self.width], image_size)

        image = self.image_stack[item]
        img_shape = self.org_img_size[item]

        if self.path_label_dir:
            label = self.label_stack[item]

        if self.aug:
            if iu.get_random() > 0.5:  # w/ 50% probability, affine transformation.
                mat1 = mu.generate_random_rotation_around_axis((1, 0, 0), 20)
                mat2 = mu.generate_random_shear(5)
                mat = mat1 * mat2

                image = ni.affine_transform(image, matrix=mat, cval=-1024.0)

                if self.path_label_dir:
                    label = ni.affine_transform(label, matrix=mat) > 0.5
                    label = label.astype(np.uint8)


        # gradient map save.
        img4grad = iu.normalize(image, -340, 360)
        # img4grad = iu.resize_image(img4grad, False, [self.depth, self.height, self.width])
        edge = iu.get_gradient_image(img4grad)

        if self.aug:
            if iu.get_random() > 0.3:  # w/ 70% probability, add gaussian noise.
                image = iu.add_gaussian_noise(image)

            if iu.get_random() > 0.2:  # w/ 80% probability, cutout augmentation.
                image = iu.cutout(image)

        image = iu.normalize(image, -340, 360)

        # edge = iu.get_gradient_image(image)

        # contour = contour.astype(np.float32) * edge
        # contour = iu.normalize_all(contour)

        if self.path_label_dir:
            contour = iu.get_edge_image(label)

            sample = {'image': image, 'img_shape':img_shape,
                      'label': label, 'edge': edge, 'contour': contour,
                      'filename': self.filename[item]}
        else:
            sample = {'filename': self.filename[item],
                      'image': image,
                      'img_shape': img_shape,
                      'edge': edge}

        return sample

        # for image_path, label_path in zip(self.image_paths, self.label_paths):
        #     # print(image_path)
        #     # print(label_path)
        #     image = read_raw(image_path, 'int16', [self.width, self.height, self.depth], image_size)
        #     label = read_raw(label_path, 'int8', [self.width, self.height, self.depth], image_size)
