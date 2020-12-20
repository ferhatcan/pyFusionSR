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
CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2ADAS_HSVsingleChannel.ini"

which_channel = 2
channel_type = 'HSV'

def imshow(imageList):
    for i in range(len(imageList)):
        imageList[i] = (np.array(imageList[i]).transpose((0, 2, 3, 1)) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate(imageList, axis=0)), grid=[len(imageList), imageList[0].shape[0] * 2], figSize=8)

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER, CONFIG_FILE_NAME)
    experiment.load('model_last')

    # rootPath = '/media/ferhatcan/common/Image_Datasets/rgbt-ped-detection/data/kaist-rgbt/images/set01/V000/'
    # rootPath = '/media/ferhatcan/common/Image_Datasets/TNO_Image_Fusion_Dataset/Triclobs_images/Bosnia/'
    rootPath = '/media/ferhatcan/common/Image_Datasets/VIFB-master/input/'
    # rootPath = '/media/ferhatcan/common/Image_Datasets/Flir/UPDATE 8-19-19_ SB Free Dataset-selected/FLIR_ADAS_1_3/val/'

    visible = Image.open(os.path.join(rootPath, 'VI/snow.jpg'))
    ir = Image.open(os.path.join(rootPath, 'IR/snow.jpg'))
    ir = ir.convert('L')
    # visible = visible.convert('RGB')
    color_im = visible.convert(channel_type)
    B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', color_im.tobytes())
    visible = Image.fromarray(B[:, :, which_channel], 'L')

    original_size = [visible.size[1], visible.size[0]]
    desiredSize = [value - (value % 32) for value in original_size]

    resize = transforms.Resize(size=desiredSize, interpolation=Image.BICUBIC)
    resize_original = transforms.Resize(size=original_size, interpolation=Image.BICUBIC)

    ir = resize(ir)
    visible = resize(visible)
    color_im = resize(color_im)
    B = np.ndarray((visible.size[1], visible.size[0], 3), 'u1', color_im.tobytes())

    # Transform to tensor
    hr_image = tvF.to_tensor(visible)
    lr_image = tvF.to_tensor(ir)

    # imshow([lr_image.unsqueeze(dim=0), lr_image.unsqueeze(dim=0)])

    data = dict()
    data["inputs"] = [lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)]


    output = experiment.inference(data)
    # data["result"] = [kn.normalize_min_max(output["results"][0])]
    data["gts"] = data["inputs"]

    # imshow([lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)])
    lr_inp = ir.convert('RGB')
    lr_inp = resize_original(lr_inp)
    lr_inp = tvF.to_tensor(lr_inp).unsqueeze(dim=0)

    vis_inp = color_im.convert('RGB')
    vis_inp = resize_original(vis_inp)
    vis_inp = tvF.to_tensor(vis_inp).unsqueeze(dim=0)
    # imshow([lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0), data["result"][0].cpu().detach()])

    data["result"] = [output["results"][0]]

    print(B[:, :, 2].max(), B[:, :, 2].min())
    print(data["result"][0].cpu().detach().numpy().max(), data["result"][0].cpu().detach().numpy().min())
    out = np.copy(B)
    out[:, :, which_channel] = data["result"][0].cpu().detach().numpy() * 255
    outPil = Image.fromarray(out, channel_type)
    outPil = outPil.convert('RGB')

    out_final = resize_original(outPil)

    out_final = tvF.to_tensor(out_final).unsqueeze(dim=0)

    imshow([lr_inp, vis_inp, out_final])
    # imshow([lr_inp, hr_image.unsqueeze(dim=0), data["result"][0].cpu().detach()])
    # imshow([kn.apply_grayscale(lr_inp), kn.apply_grayscale(hr_image.unsqueeze(dim=0)), kn.apply_grayscale(data["result"][0].cpu().detach())])


    # data["result"][0] = data["gts"][0]

    benchmarkResults = experiment.benchmark.getBenchmarkResults(data)
    for bench in benchmarkResults:
        for scoreTxt in benchmarkResults[bench]['score_texts']:
            print(scoreTxt)

    # lr_inp = torch.cat((data["edges"]["gts"][0].cpu().detach(), data["edges"]["gts"][0].cpu().detach(), data["edges"]["gts"][0].cpu().detach()), dim=1)
    # imshow([lr_inp, data["edges"]["gts"][1].cpu().detach()])
    imshow([data["edges"]["result"][0].cpu().detach(),  data["edges"]["gts"][0].cpu().detach(), data["edges"]["gts"][1].cpu().detach(), data["result"][0].cpu().detach()])
    # tmp = 0

if __name__ == '__main__':
    main()