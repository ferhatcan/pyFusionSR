import torch
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import functional as tvF
import random
import PIL.Image as Image
from PIL import ImageFilter

from dataloaders.baseDataset import BaseDataset

class KaistDataset(BaseDataset):
    def __init__(self, args, train=True):
        self.args = args
        ds_files = self.args.train_set_paths if train else self.args.test_set_paths
        self.root = os.path.split(os.path.split(ds_files[0])[0])[0]
        self.root = os.path.join(self.root, 'images')
        super(KaistDataset, self).__init__([], args)

        self.imageFiles = np.array([np.array([]), np.array([])])
        self.extract_image_files(ds_files)


    def extract_image_files(self, ds_files):
        lwir_im_paths = self.imageFiles[0]
        visible_im_paths = self.imageFiles[1]
        for ds_file in ds_files:
            im_file = open(ds_file, 'r')
            lines = im_file.readlines()
            im_file.close()

            for line in lines:
                if line[-1:] == '\n':
                    line_split = os.path.split(line[:-1])
                else:
                    line_split = os.path.split(line)
                lwir_im_path = os.path.join(self.root, line_split[0], 'lwir', line_split[1] + '.jpg')
                visible_im_path = os.path.join(self.root, line_split[0], 'visible', line_split[1] + '.jpg')
                lwir_im_paths = np.append(lwir_im_paths, np.array([lwir_im_path]))
                visible_im_paths = np.append(visible_im_paths, np.array([visible_im_path]))
                # lwir_im_paths.append(lwir_im_path)
                # visible_im_paths.append(visible_im_path)

        #print(lwir_im_paths[0], len(lwir_im_paths))
        #print(visible_im_paths[0], len(visible_im_paths))

        self.imageFiles = np.array([lwir_im_paths, visible_im_paths])

    def __len__(self):
        return self.imageFiles.shape[1]

    def __getitem__(self, index: int) -> dict:
        hr_image_lwir = Image.open(self.imageFiles[0][index])
        hr_image_visible = Image.open(self.imageFiles[1][index])
        lr_ir, hr_ir, lr_eo, hr_eo = self.transform_multi(hr_image_lwir, hr_image_visible)
        return self.fillOutputDataDict([lr_ir, lr_eo], [hr_ir, hr_eo])

    def transform(self, image):
        hr_image = image
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
        if self.include_noise:
            lr_image = np.array(np.clip((lr_image +
                                         np.random.normal(self.noise_mean, self.noise_sigma, lr_image.shape)),
                                        a_min=0, a_max=255).astype("uint8"))

        # desired channel number should be checked
        if self.channel_number == 3 and lr_image.shape[-1] == 1:
            lr_image = np.stack([lr_image[np.newaxis, ...]] * 3, axis=0)
            hr_image = np.stack([hr_image[np.newaxis, ...]] * 3, axis=0)

        # Transform to tensor
        hr_image = tvF.to_tensor(Image.fromarray(hr_image))
        lr_image = tvF.to_tensor(Image.fromarray(lr_image))

        # # apply normalization
        # if self.normalize == "zeroMean":
        #     # todo Mean & STD of the dataset should be given or It can be calculated in a method
        #     hr_means = [hr_image.mean() for i in range(hr_image.shape[0])]
        #     lr_means = [lr_image.mean() for i in range(lr_image.shape[0])]
        #     hr_stds = [hr_image.std() for i in range(hr_image.shape[0])]
        #     lr_stds = [lr_image.std() for i in range(lr_image.shape[0])]
        #     hr_image = tvF.normalize(hr_image, hr_means, hr_stds)
        #     lr_image = tvF.normalize(lr_image, lr_means, lr_stds)
        # elif self.normalize == "between01":
        #     hr_mins = [hr_image.min() for i in range(hr_image.shape[0])]
        #     lr_mins = [lr_image.min() for i in range(lr_image.shape[0])]
        #     hr_ranges = [hr_image.max() - hr_image.min() for i in range(hr_image.shape[0])]
        #     lr_ranges = [lr_image.max() - lr_image.min() for i in range(lr_image.shape[0])]
        #     hr_image = tvF.normalize(hr_image, hr_mins, hr_ranges)
        #     lr_image = tvF.normalize(lr_image, lr_mins, lr_ranges)

        # apply normalization
        if self.normalize == "zeroMean":
            # todo Mean & STD of the dataset should be given or It can be calculated in a method
            hr_means = [hr_image.mean() for i in range(hr_image.shape[0])]
            lr_means = [lr_image.mean() for i in range(lr_image.shape[0])]
            hr_stds = [hr_image.std() for i in range(hr_image.shape[0])]
            lr_stds = [lr_image.std() for i in range(lr_image.shape[0])]
            if hr_stds[0].item() == 0 or lr_stds[0].item() == 0:
                hr_image = tvF.normalize(hr_image, hr_means, [1, ])
                lr_image = tvF.normalize(lr_image, lr_means, [1, ])
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
                # lr_image = tvF.normalize(lr_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            else:
                hr_image = tvF.normalize(hr_image, hr_mins, [1, ])
                lr_image = tvF.normalize(lr_image, lr_mins, [1, ])
        elif self.normalize == "divideBy255":
            hr_image = tvF.normalize(hr_image, [0, ], [1, ])
            lr_image = tvF.normalize(lr_image, [0, ], [1, ])

        # if self.channel_number == 3 & hr_image.size[-1] == 1:
        #     hr_image = hr_image
        #     lr_image = lr_image

        return lr_image, hr_image

#Testing purposes
# from options import options
# CONFIG_FILE_NAME = "../configs/encoderDecoderFusionv2ADAS_HSVsingleChannel.ini"
# args = options(CONFIG_FILE_NAME)
# kaist = KaistDataset(args.argsDataset)
# kaist.hr_shape = [256, 256]
# kaist.lr_shape = [256, 256]
# print(len(kaist))
# data = kaist.__getitem__(random.randint(0, len(kaist)))
#
# print(data['gts'][1].numpy().transpose((1, 2, 0)).squeeze().max(), data['gts'][1].numpy().transpose((1, 2, 0)).squeeze().min())
# import matplotlib.pyplot as plt
# plt.ion()
#
# tmp = data['inputs'][1].numpy().transpose((1, 2, 0)).squeeze()
# plt.imshow(data['inputs'][1].numpy().transpose((1, 2, 0)).squeeze(), cmap='gray')
# plt.waitforbuttonpress()
# plt.figure()
# plt.imshow(data['gts'][0].numpy().transpose((1, 2, 0)).squeeze(), cmap='gray')
# plt.waitforbuttonpress()
#tmp = 0