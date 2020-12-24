import torch
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import functional as tvF
import random
import PIL.Image as Image
from PIL import ImageFilter
import cv2
import imutils

from dataloaders.IDataLoader import IDataLoader

class FlirAdasDataset(IDataLoader):
    def __init__(self, args, train=True):
        super(FlirAdasDataset, self).__init__()

        self.image_paths = args.train_set_paths if train else args.test_set_paths

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
        assert self.channel_type in possible_channel_types, "Given channel type must be included in {}".format(possible_channel_types)

        if self.downgrade == "bicubic":
            self.downgrade = Image.BILINEAR
        elif self.downgrade == "nearest":
            self.downgrade = Image.NEAREST
        elif self.downgrade == "bilinear":
            self.downgrade = Image.BILINEAR

        self.lr_shape = [i // self.scale for i in self.hr_shape]
        self.irPath = 'thermal_8_bit'
        self.visPath = 'RGB'
        self.extensions = ["jpg", "jpeg", "png"]

        # self.imageFiles = dict()
        # self.imageFiles["thermal"] = np.array([], dtype=object)
        # self.imageFiles["visible"] = np.array([], dtype=object)
        self.imageFiles = np.array([np.array([]), np.array([])])

        self.extract_image_files()
        assert (self.imageFiles.shape[1] > 0), "There should be an valid dataset path"

        self.homography = np.load("./dataloaders/homography_flirADAS.npy")

    def extract_image_files(self):
        lwir_im_paths = self.imageFiles[0]
        visible_im_paths = self.imageFiles[1]
        for image_path in self.image_paths:
            searchPath = os.path.join(image_path, self.visPath)
            irSearchPath = os.path.join(image_path, self.irPath)
            for file in os.listdir(searchPath):
                if os.path.isfile(os.path.join(searchPath, file)):
                    ext = file.split('.')[-1]
                    if ext in self.extensions:
                        if os.path.exists(os.path.join(irSearchPath,  file.split('.')[0] + '.jpeg')):
                            visible_im_paths = np.append(visible_im_paths, np.array([os.path.join(searchPath, file)]))
                            lwir_im_paths = np.append(lwir_im_paths, np.array([os.path.join(irSearchPath, file.split('.')[0] + '.jpeg')]))

        self.imageFiles = np.array([lwir_im_paths, visible_im_paths])

    def __len__(self):
        return self.imageFiles.shape[1]

    def __getitem__(self, item: int) -> dict:
        visIm = Image.open(self.imageFiles[1][item]).convert('RGB')
        irIm = Image.open(self.imageFiles[0][item]).convert('RGB')

        visIm = np.array(visIm)
        irIm = np.array(irIm)

        irIm = cv2.resize(irIm, (visIm.shape[1], visIm.shape[0]))

        height, width = irIm.shape[0], irIm.shape[1]

        if height * width < 1e6:
            offset = 50
        else:
            offset = 200

        irIm = cv2.warpPerspective(irIm, self.homography, (width, height))

        # DEBUGGING
        # dst = cv2.addWeighted(irIm, 0.7, visIm, 0.3, 0.0)
        # cv2.imshow("Blended", imutils.resize(dst, width=1000))
        # cv2.waitKey()

        ir_lr, ir_hr, vis_lr, vis_hr = self.transform_multi(Image.fromarray(irIm[offset:height-offset, offset:width-offset]),
                                                            Image.fromarray(visIm[offset:height-offset, offset:width-offset]))

        return self.fillOutputDataDict([ir_lr, vis_lr], [ir_hr, vis_hr])

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

        # print(np.array(hr_image).max(), np.array(hr_image).min())
        # print(np.array(hr_image2).max(), np.array(hr_image2).min())

        return [*self.transform(hr_image), *self.transform(hr_image2)]

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
            else:
                hr_image = tvF.normalize(hr_image, hr_mins, [1, ])
                lr_image = tvF.normalize(lr_image, lr_mins, [1, ])
        elif self.normalize == "divideBy255":
            hr_image = tvF.normalize(hr_image, [0, ], [1, ])
            lr_image = tvF.normalize(lr_image, [0, ], [1, ])

        return lr_image, hr_image

    @staticmethod
    def randomGenerator():
        return random.random(), random.random()


#Testing purposes
# from utils.checkpoint import checkpoint
# from options import options
# CONFIG_FILE_NAME = "../configs/encoderDecoderFusionv2ADAS_HSVsingleChannel.ini"
# args = options(CONFIG_FILE_NAME)
# adas = FlirAdasDataset(args.argsDataset)
# adas.hr_shape = [512, 512]
# adas.lr_shape = [512, 512]
# print(len(adas))
# data = adas.__getitem__(random.randint(0, len(adas)))
# # data = adas.__getitem__(2)
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

# tmp = 0