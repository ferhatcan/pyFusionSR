import PIL.Image as Image
from skimage import color
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

which_channel = 0
channel_type = 'YCbCr'

def imshow(imageList):
    for i in range(len(imageList)):
        imageList[i] = (np.array(imageList[i]).transpose((0, 2, 3, 1)) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate(imageList, axis=0)), grid=[len(imageList), imageList[0].shape[0] * 2], figSize=8)

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER, CONFIG_FILE_NAME)
    experiment.load('model_last')

    # validation_loss, validation_benchmark, _ = experiment.test_dataloader(experiment.dataloaders["validation"])

    # rootPath = '/media/ferhatcan/common/Image_Datasets/rgbt-ped-detection/data/kaist-rgbt/images/set01/V000/'
    # rootPath = '/media/ferhatcan/common/Image_Datasets/TNO_Image_Fusion_Dataset/Triclobs_images/Bosnia/'
    rootPath = '/media/ferhatcan/common/Image_Datasets/VIFB-master/input/'
    # rootPath = '/media/ferhatcan/common/Image_Datasets/Flir/UPDATE 8-19-19_ SB Free Dataset-selected/FLIR_ADAS_1_3/val/'

    visible = Image.open(os.path.join(rootPath, 'VI/carWhite.jpg'))
    ir = Image.open(os.path.join(rootPath, 'IR/carWhite.jpg'))
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

    hr_image = hr_image * 2 - 1
    lr_image = lr_image * 2 - 1

    imshow([(lr_image.unsqueeze(dim=0) + 1) * 0.5, (hr_image.unsqueeze(dim=0) + 1) * 0.5])

    data = dict()
    data["inputs"] = [lr_image.unsqueeze(dim=0), hr_image.unsqueeze(dim=0)]


    output = experiment.inference(data)
    # data["result"] = [torch.clamp(output["results"][0], 0, 1)]
    data["result"] = [(output["results"][0] + 1) * 0.5]
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


    # data["result"][0] = data["result"][0] * 255

    print(B[:, :, which_channel].max(), B[:, :, which_channel].min())
    print(data["result"][0].cpu().detach().numpy().max(), data["result"][0].cpu().detach().numpy().min())
    out = np.copy(B)
    out[:, :, which_channel] = data["result"][0].cpu().squeeze().detach().numpy() * 255
    # print(out[:, :, 2].max(), out[:, :, 2].min())
    # out_np = np.array(out, dtype=np.uint8)
    outPil = Image.fromarray(out, channel_type)
    outPil = outPil.convert('RGB')

    # inpImage = Image.open(os.path.join(rootPath, 'VI/labMan.jpg')).convert('RGB')
    # np_inpImage = np.array(inpImage, dtype=np.float32) / 255
    # B_lab = color.rgb2lab(np_inpImage)
    #
    # B_lab[:, :, 0] = 100

    # print(B_lab[:, :, 0].max(), B_lab[:, :, 0].min())
    # B_rgb = color.lab2rgb(B_lab)
    # print(B_rgb[:, :, 0].max(), B_rgb[:, :, 0].min())
    # outPil = Image.fromarray((B_rgb[:, :, 0] * 255).astype(np.uint8), 'L').convert('RGB')

    out_final = resize_original(outPil)

    # imshow([tvF.to_tensor(out_final.convert('L')).unsqueeze(dim=0), (lr_image.unsqueeze(dim=0) + 1) * 0.5, (hr_image.unsqueeze(dim=0) + 1) * 0.5, data["result"][0].cpu().detach()])

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