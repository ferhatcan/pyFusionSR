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


def imshow(imageList):
    for i in range(len(imageList)):
        imageList[i] = (np.array(imageList[i]).transpose((0, 2, 3, 1)) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate(imageList, axis=0)), grid=[len(imageList), imageList[0].shape[0] * 2], figSize=10)

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER)
    experiment.load('model_best')

    rootPath = '/media/ferhatcan/common/Image_Datasets/TNO_Image_Fusion_Dataset/Athena_images/soldier_behind_smoke_3/'
    # rootPath = '/media/ferhatcan/common/Image_Datasets/Flir/UPDATE 8-19-19_ SB Free Dataset-selected/FLIR_ADAS_1_3/val/'

    visible = Image.open(os.path.join(rootPath, 'VIS_meting012-1700_r.bmp'))
    ir = Image.open(os.path.join(rootPath, 'IR_meting012-1700_g.bmp'))
    ir = ir.convert('L')
    visible = visible.convert('RGB')

    resize = transforms.Resize(size=[480, 640], interpolation=Image.BICUBIC)
    ir = resize(ir)
    visible = resize(visible)

    # Transform to tensor
    hr_image = tvF.to_tensor(visible)
    lr_image = tvF.to_tensor(ir)

    data = dict()
    data["inputs"] = [lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)]

    output = experiment.inference(data)
    data["result"] = [kn.normalize_min_max(output["results"][0])]
    data["gts"] = data["inputs"]

    # imshow([lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)])
    lr_inp = torch.cat((lr_image.unsqueeze(dim=0), lr_image.unsqueeze(dim=0), lr_image.unsqueeze(dim=0)), dim=1)
    imshow([lr_inp, lr_inp, data["result"][0].cpu().detach()])

    # data["result"][0] = data["gts"][0]

    benchmarkResults = experiment.benchmark.getBenchmarkResults(data)
    for bench in benchmarkResults:
        for scoreTxt in benchmarkResults[bench]['score_texts']:
            print(scoreTxt)

    lr_inp = torch.cat((data["edges"]["gts"][0].cpu().detach(), data["edges"]["gts"][0].cpu().detach(), data["edges"]["gts"][0].cpu().detach()), dim=1)
    imshow([lr_inp, data["edges"]["gts"][1].cpu().detach()])
    imshow([data["edges"]["result"][0].cpu().detach(),  data["result"][0].cpu().detach()])
    # tmp = 0

if __name__ == '__main__':
    main()