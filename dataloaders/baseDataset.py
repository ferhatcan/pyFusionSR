# Author: @ferhatcan
# Date: 24/04/20

import torch
import numpy as np
import math
from torchvision import transforms
from torchvision.transforms import functional as tvF
import os
import random
import PIL.Image as Image
from PIL import ImageFilter

from dataloaders.IDataLoader import IDataLoader

# This base dataset loader class will implement the following properties
# +1: Train dataloader & Validation DataLoader
# +2: Adding Noise
# +3: Adding Blur
# +4: Downgrade options
# +5: Transforms
# +6: Random Crops or ..
# 7:

class BaseDataset(IDataLoader):
    def __init__(self, image_paths: list, args):
        super(BaseDataset, self).__init__()

        self.image_paths = image_paths
        self.scale = args.scale
        self.include_noise = args.include_noise
        self.noise_sigma = args.noise_sigma
        self.noise_mean = args.noise_mean
        self.include_blur = args.include_blur
        self.blur_radius = args.blur_radius
        self.normalize = args.normalize
        self.random_flips = args.random_flips
        self.channel_number = args.channel_number
        self.hr_shape = args.hr_shape
        self.downgrade = args.downgrade
        self.channel_type = args.channel_type

        possible_channel_types = ["RGB", "YCbCr", "HSV", "L"]
        assert self.channel_type in possible_channel_types, "Given channel type must be included in {}".format(
            possible_channel_types)

        if self.downgrade == "bicubic":
            self.downgrade = Image.BILINEAR
        elif self.downgrade == "nearest":
            self.downgrade = Image.NEAREST
        elif self.downgrade == "bilinear":
            self.downgrade = Image.BILINEAR

        self.lr_shape = [i // self.scale for i in self.hr_shape]

        self.extensions = ["jpg", "jpeg", "png"]
        self.imageFiles = []
        for image_path in self.image_paths:
            for root, directory, fileNames in os.walk(image_path):
                for file in fileNames:
                    ext = file.split('.')[-1]
                    if ext in self.extensions:
                        self.imageFiles.append(os.path.join(root, file))

        if not self.image_paths == []:
            assert (len(image_paths) > 0), "There should be an valid image path"

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, item: int) -> dict:
        data = dict()
        hr_image = Image.open(self.imageFiles[item])
        lr_image, hr_image = self.transform(hr_image)
        return self.fillOutputDataDict([lr_image], [hr_image])

    @staticmethod
    def fillOutputDataDict(inputs: list, gts: list) -> dict:
        data = dict()
        data["inputs"] = inputs
        data["gts"] = gts
        return data

    def transform_multi(self, image_ir, image_visible):
        image_ir = image_ir.convert('L')
        if self.channel_number == 3:
            image_visible = image_visible.convert('RGB')
        else:
            if self.channel_type == "YCbCr":
                ycbcr = image_visible.convert('YCbCr')
                B = np.ndarray((image_visible.size[1], image_visible.size[0], 3), 'u1', ycbcr.tobytes())
                image_visible = Image.fromarray(B[:, :, 0], 'L')
            elif self.channel_type == "HSV":
                hsv = image_visible.convert('HSV')
                B = np.ndarray((image_visible.size[1], image_visible.size[0], 3), 'u1', hsv.tobytes())
                image_visible = Image.fromarray(B[:, :, -1], 'L')
            else:
                image_visible = image_visible.convert('L')
        # make image dimensions same to make crop right segments as possible
        # resize = transforms.Resize(size=[image_ir.height, image_ir.width], interpolation=self.downgrade)
        # image_visible = resize(image_visible)
        # Resize input image if its dimensions smaller than desired dimensions
        resize = transforms.Resize(size=self.hr_shape, interpolation=self.downgrade)
        # if not (image_ir.width > self.hr_shape[0] and image_ir.height > self.hr_shape[1]):
        #     image_ir = resize(image_ir)
        # if not (image_visible.width > self.hr_shape[0] and image_visible.height > self.hr_shape[1]):
        #     image_visible = resize(image_visible)

        # random crop
        # crop = transforms.RandomCrop(size=self.hr_shape)
        # i, j, h, w = crop.get_params(image_ir, self.hr_shape)
        # hr_image = tvF.crop(image_ir, i, j, h, w)
        # hr_image2 = tvF.crop(image_visible, i, j, h, w)

        hr_image = resize(image_ir)
        hr_image2 = resize(image_visible)

        return [*self.transform(hr_image), *self.transform(hr_image2)]

    def transform(self, image):
        if self.channel_number == 1:
            image = image.convert('L')

        # Resize input image if its dimensions smaller than desired dimensions
        resize = transforms.Resize(size=self.hr_shape, interpolation=self.downgrade)
        if not (image.width > self.hr_shape[0] and image.height > self.hr_shape[1]):
            image = resize(image)


        # random crop
        crop = transforms.RandomCrop(size=self.hr_shape)
        hr_image = crop(image)

        # downscale to obtain low-resolution image
        resize = transforms.Resize(size=self.lr_shape, interpolation=self.downgrade)
        lr_image = resize(hr_image)

        # apply blur
        if self.include_blur:
            lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # apply random transforms
        if self.random_flips:
            horiz_random, vert_random = self.randomGenerator()
            # random horizontal flip
            if horiz_random > 0.5:
                hr_image = tvF.hflip(hr_image)
                lr_image = tvF.hflip(lr_image)

            # random vertical flip
            if vert_random > 0.5:
                hr_image = tvF.vflip(hr_image)
                lr_image = tvF.vflip(lr_image)


        # apply noise
        lr_image = np.array(lr_image)
        hr_image = np.array(hr_image)

        # if (hr_image.max() - hr_image.min()) == 0 or (lr_image.max() - lr_image.min()) == 0:
        #     print('zero image')

        if self.include_noise:
            lr_image = np.array(np.clip((lr_image +
                                         np.random.normal(self.noise_mean, self.noise_sigma, lr_image.shape)),
                                        a_min=0, a_max=255).astype("uint8"))

        self.check_nan(lr_image)
        self.check_nan(hr_image)

        # desired channel number should be checked
        if self.channel_number == 3 and lr_image.shape[-1] == 1:
            lr_image = np.stack([lr_image[np.newaxis, ...]] * 3, axis=0)
            hr_image = np.stack([hr_image[np.newaxis, ...]] * 3, axis=0)

        # Transform to tensor
        hr_image = tvF.to_tensor(Image.fromarray(hr_image))
        lr_image = tvF.to_tensor(Image.fromarray(lr_image))

        # apply normalization
        if self.normalize == "zeroMean":
            # todo Mean & STD of the dataset should be given or It can be calculated in a method
            hr_means = [hr_image.mean() for i in range(hr_image.shape[0])]
            lr_means = [lr_image.mean() for i in range(lr_image.shape[0])]
            hr_stds = [hr_image.std() for i in range(hr_image.shape[0])]
            lr_stds = [lr_image.std() for i in range(lr_image.shape[0])]
            if hr_stds[0].item() == 0 or lr_stds[0].item() == 0:
                hr_image = tvF.normalize(hr_image, hr_means, [1, 1, 1])
                lr_image = tvF.normalize(lr_image, lr_means, [1, 1, 1])
            else:
                hr_image = tvF.normalize(hr_image, hr_means, hr_stds)
                lr_image = tvF.normalize(lr_image, lr_means, lr_stds)
        elif self.normalize == "between01":
            hr_mins = [hr_image.min() for i in range(hr_image.shape[0])]
            lr_mins = [lr_image.min() for i in range(lr_image.shape[0])]
            hr_ranges = [hr_image.max() - hr_image.min() for i in range(hr_image.shape[0])]
            lr_ranges = [lr_image.max() - lr_image.min() for i in range(lr_image.shape[0])]
            if not (hr_ranges[0].item() == 0 or lr_ranges[0].item() == 0):
                hr_image = tvF.normalize(hr_image, hr_mins, hr_ranges)
                lr_image = tvF.normalize(lr_image, lr_mins, lr_ranges)
                # hr_image = tvF.normalize(hr_image, [0.5,], [0.5,])
                #lr_image = tvF.normalize(lr_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            else:
                hr_image = tvF.normalize(hr_image, hr_mins, [1, 1, 1])
                lr_image = tvF.normalize(lr_image, lr_mins, [1, 1, 1])
            #     print('image not correct')
        elif self.normalize == "divideBy255":
            hr_image = tvF.normalize(hr_image, [0, ], [1, ])
            lr_image = tvF.normalize(lr_image, [0, ], [1, ])

        # if self.channel_number == 3 & hr_image.size[-1] == 1:
        #     hr_image = hr_image
        #     lr_image = lr_image
        self.check_nan(lr_image)
        self.check_nan(hr_image)

        return lr_image, hr_image

    @staticmethod
    def randomGenerator():
        return random.random(), random.random()

    @staticmethod
    def check_nan(inp):
        if torch.is_tensor(inp):
            dummy = torch.ones(*inp.shape[1:])
            for i in range(inp.shape[0]):
                if torch.sum(torch.isnan(inp[i, ...])):
                    inp[i, ...] = dummy
        else:
            dummy = np.ones(inp.shape)
            for i in range(inp.shape[0]):
                if np.sum(np.isnan(inp[i, ...])):
                    inp[i, ...] = dummy

# Test dataset class

# validation_split = 0.1
# batch_size = 4
# dataset_path = "/home/ferhatcan/Image_Datasets/ir_sr_challange"
# shuffle_dataset = True
# hr_shape = [360, 640]
#
# ir_challange_dataset = BaseDatasetLoader([dataset_path],
#                                          scale=2, normalize="between01",
#                                          hr_shape=hr_shape,
#                                          randomflips=False)
# dataset_size = len(ir_challange_dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset:
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
#
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
# validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
#
# train_loader = torch.utils.data.DataLoader(ir_challange_dataset, batch_size=batch_size,
#                                            sampler=train_sampler)
# validation_loader = torch.utils.data.DataLoader(ir_challange_dataset, batch_size=batch_size,
#                                            sampler=validation_sampler)
#
# from utils.visualization import imshow_image_grid
# data = next(iter(train_loader))
# #for i, data in enumerate(train_loader):
#     #print(data[0].size(), data[1].size())
# lr_batch = np.array(data[0]).transpose(0, 2, 3, 1)
# lr_batch_SR = [tvF.resize(Image.fromarray(lr_batch[i,...].squeeze()), size=hr_shape, interpolation=Image.BICUBIC)
#                         for i in range(lr_batch.shape[0])]
# lr_batch = [np.array(lr_batch_SR[i])[..., np.newaxis] for i in range(len(lr_batch_SR))]
# lr_batch = np.stack(lr_batch, axis=0)
# hr_batch = np.array(data[1]).transpose(0, 2, 3, 1)
# imshow_image_grid(np.array(np.concatenate([lr_batch, hr_batch], axis=0)), grid=[2, 4], figSize=10)
#
# tmp = 0
