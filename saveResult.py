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

DESIRED_ASSEMBLER = "fusionv2ADAS"
CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2ADAS_YsingleChannels.ini"
ROOT_PATH = "/media/ferhatcan/common/Image_Datasets/VIFB-master/input/"
OUTPUT_PATH = "./Outputs/IV_images/"

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
    for visible_filename, ir_filename in zip(imageFiles["visible"], imageFiles["thermal"]):
        visible = Image.open(visible_filename)
        ir = Image.open(ir_filename)

        ir = ir.convert('L')
        # visible = visible.convert('RGB')
        ycbcr = visible.convert('YCbCr')
        B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', ycbcr.tobytes())
        visible = Image.fromarray(B[:, :, 0], 'L')

        resize = transforms.Resize(size=[480, 640], interpolation=Image.BICUBIC)
        ir = resize(ir)
        visible = resize(visible)
        ycbcr = resize(ycbcr)
        B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', ycbcr.tobytes())

        # Transform to tensor
        hr_image = tvF.to_tensor(visible)
        lr_image = tvF.to_tensor(ir)

        data = dict()
        data["inputs"] = [lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)]

        output = experiment.inference(data)
        data["result"] = [kn.normalize_min_max(output["results"][0])]

        out = np.copy(B)
        out[:, :, 0] = data["result"][0].cpu().detach().numpy() * (235 - 16) + 16
        outPil = Image.fromarray(out, 'YCbCr')
        outPil = outPil.convert('RGB')

        saveName = os.path.split(ir_filename)[-1]
        saveName, ext = saveName.split('.')
        saveName = saveName + '_fused.' + ext

        os.makedirs(OUTPUT_PATH, exist_ok=True)

        outPil.save(os.path.join(OUTPUT_PATH, saveName))

if __name__ == '__main__':
    main()