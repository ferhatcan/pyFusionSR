"""
Give input folder with:
    RGB:
    IR:
Results:
    TestName
        ExperimentName:
"""

import PIL.Image as Image
from torchvision.transforms import functional as tvF
import torchvision.transforms as transforms
import torch
import kornia as kn
import numpy as np
from utils.visualization import imshow_image_grid
from utils.benchmark import BenchmarkMetrics
import os
from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

DESIRED_ASSEMBLER = "fusionv2ADASKAIST"
CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2ADAS+KAIST_YsingleChannels.ini"
ROOT_PATH = "/media/ferhatcan/common/Image_Datasets/VIFB-master/input/"
OUTPUT_PATH = "./Outputs/Y+IR-ADAS_KAIST_norm1-1_onlyQEQ/"

which_channel = 0
channel_type = 'YCbCr'

def extract_image_files():
    extensions = ["jpg", "jpeg", "png"]

    imageFiles = dict()
    imageFiles["thermal"] = []
    imageFiles["visible"] = []

    searchPath = os.path.join(ROOT_PATH, "VI")
    irSearchPath = os.path.join(ROOT_PATH, "IR")
    for file in os.listdir(searchPath):
        if os.path.isfile(os.path.join(searchPath, file)):
            ext = file.split('.')[-1]
            if ext in extensions:
                if os.path.exists(os.path.join(irSearchPath, file)):
                    imageFiles["visible"].append(os.path.join(searchPath, file))
                    imageFiles["thermal"].append(os.path.join(irSearchPath, file))
    return imageFiles

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER, CONFIG_FILE_NAME)
    experiment.load('model_last')

    imageFiles = extract_image_files()
    os.makedirs(OUTPUT_PATH, exist_ok=True)


    for visible_filename, ir_filename in zip(imageFiles["visible"], imageFiles["thermal"]):
        visible = Image.open(visible_filename)
        ir = Image.open(ir_filename)

        ir = ir.convert('L')
        # visible = visible.convert('RGB')
        ycbcr = visible.convert(channel_type)
        B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', ycbcr.tobytes())
        visible = Image.fromarray(B[:, :, which_channel], 'L')

        original_size = [visible.size[1], visible.size[0]]
        desiredSize = [value - (value % 32) for value in original_size]

        resize = transforms.Resize(size=desiredSize, interpolation=Image.BICUBIC)
        ir = resize(ir)
        visible = resize(visible)
        ycbcr = resize(ycbcr)
        B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', ycbcr.tobytes())

        # Transform to tensor
        hr_image = tvF.to_tensor(visible)
        lr_image = tvF.to_tensor(ir)

        hr_image = hr_image * 2 - 1
        lr_image = lr_image * 2 - 1

        data = dict()
        data["inputs"] = [lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)]

        output = experiment.inference(data)
        # data["result"] = [kn.normalize_min_max(output["results"][0])]
        data["result"] = [(output["results"][0] + 1) * 0.5]
        print(B[:, :, which_channel].max(), B[:, :, which_channel].min())
        print(data["result"][0].cpu().detach().numpy().max(), data["result"][0].cpu().detach().numpy().min())
        # print(data["inputs"][0].cpu().detach().numpy().max(), data["inputs"][0].cpu().detach().numpy().min())
        out = np.copy(B)
        out[:, :, which_channel] = data["result"][0].cpu().detach().numpy() * 255
        outPil = Image.fromarray(out, channel_type)
        outPil = outPil.convert('RGB')

        resize_original = transforms.Resize(size=original_size, interpolation=Image.BICUBIC)
        out_final = resize_original(outPil)

        saveName = os.path.split(ir_filename)[-1]
        saveName, ext = saveName.split('.')
        saveName = saveName + '_proposed.' + ext

        out_final.save(os.path.join(OUTPUT_PATH, saveName))

if __name__ == '__main__':
    main()