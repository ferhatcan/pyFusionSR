import torch
import torchvision
import torch.optim as optim
import numpy as np

from options import options
from utils.checkpoint import checkpoint
from utils.visualization import imshow_image_grid
from utils.visualization import psnr, ssim
from utils.helper_functions import *
from utils.logger import LoggerTensorBoard
from utils.benchmark import BenchmarkMetrics

from dataloaders.dataloaderGetters import *
import models.encoderdecoder as model
from loss.lossGetters import *
from methods.methodGetters import *
from optimizers.optimizerSchedulerGetters import *



from torch.utils.tensorboard import SummaryWriter
import datetime

CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2.ini"

args = options(CONFIG_FILE_NAME)
print("The system will use following resource: {:}".format(args.argsCommon.device))
print("Experiment Name: " + args.argsCommon.experiment_name)
print("Experiment will be saved to " + args.argsCommon.experiment_save_path)

# torch.manual_seed(args.seed)


def imshow(imageList):
    lr_batch = [torch.nn.functional.interpolate(imageList[0][i, ...].unsqueeze(0), scale_factor=args.argsCommon.hr_shape[0]/imageList[0].shape[2],
                                       mode='bicubic', align_corners=True).squeeze(dim=0) for i in range(imageList[0].shape[0])]
    lr_batch = torch.stack(lr_batch, dim=0)
    lr_batch = (np.array(lr_batch).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    hr_batch = (np.array(imageList[1]).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate([lr_batch, hr_batch], axis=0)), grid=[2, hr_batch.shape[0]], figSize=10)


def main():
    dataloaders = getKaistDataLoaders(args.argsDataset)
    currentModel = model.make_model(args.argsModel)
    data = next(iter(dataloaders["train"]))

    model_state_dict = currentModel.state_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")
    currentModel.to(device)
    for part in data:
        for i, image in enumerate(data[part]):
            data[part][i] = image.to(device)

    # imshow([data[0], data[1]])
    loggerTest(currentModel, data)

    result = currentModel(data["inputs"])
    data["result"] = [result]
    benchmark = BenchmarkMetrics(args.argsBenchmark)
    benchmarkResults = benchmark.getBenchmarkResults(data)
    for bench in benchmarkResults:
        for scoreTxt in benchmarkResults[bench]['score_texts']:
            print(scoreTxt)

    # imshow([data[2], result.detach()])

    possibles = globals().copy()
    loss_dict = dict()
    loss_dict["types"] = []
    loss_dict["functions"] = []
    loss_dict["weights"] = []
    for loss in args.argsLoss.loss.split('+'):
        weight, loss_type = loss.split('*')
        loss_dict["functions"].append(possibles.get('get'+loss_type+'Loss')(args.argsLoss))
        loss_dict["weights"].append(float(weight))
        loss_dict["types"].append(loss_type)
    losses = []
    for weight, loss_function in zip(loss_dict["weights"], loss_dict["functions"]):
        results = loss_function(data)
        losses.append(weight * sum(results) / len(results))
    losses.append(sum(losses))
    loss_dict["types"].append('Total')
    loss_dict["weights"].append(1)
    for i in range(len(losses)):
        print('{} loss --> [{:.3f}]'.format(loss_dict["types"][i], losses[i].item()), end='. ')
    print()

    grayscaleImList = convert2grayscale([data["inputs"][1].cpu().detach()])
    grayscaleImList.append(data["inputs"][0].cpu().detach())
    # imshow(grayscaleImList)


    lr_scheduler, optimizer = possibles.get('get'+args.argsMethod.optimizer+'Optimizer')(currentModel.parameters(), args.argsMethod)

    method = getBaseMethod(currentModel, loss_dict, optimizer, args.argsMethod)

    result, losses_ = method.validate(data)

    for i in range(len(losses_)):
        print('{} loss --> [{:.3f}]'.format(loss_dict["types"][i], losses_[i].item()), end='. ')
    print()

    print('{:} dataset size for training'.format(len(dataloaders["train"])))

    for batch_number, images in enumerate(dataloaders["train"]):
        for part in images:
            for i, image in enumerate(images[part]):
                images[part][i] = image.to(device)
        result, losses_ = method.train(images)
        if batch_number % 20:
            for i in range(len(losses_)):
                print('{} loss --> [{:.3f}]'.format(loss_dict["types"][i], losses_[i].item()), end='. ')
            print()
            break
    lr_scheduler.step()  # it should be called after each epoch

    result_validate, losses_ = method.train(data)
    result_validate, losses_ = method.validate(data)
    result_test = method.test(data)
    data["result"] = [result_test]


    for i in range(len(losses_)):
        print('{} loss --> [{:.3f}]'.format(loss_dict["types"][i], losses_[i].item()), end='. ')
    print()
    isEq = sum(getL1Loss(args.argsLoss)(data))
    print('This should be False. Because after weights updated, results should be different. --> Result [{}]'.format((isEq <= 0.01).item()))
    isEq = sum(getL1Loss(args.argsLoss)(data))
    print('This should be True. Because validation and test step should give same result for same network model. --> Result [{}]'.format((isEq <= 0.01).item()))

    # imshow([result_validate.cpu().detach(), result_test.cpu().detach()])


    tmp = 0

def loggerTest(my_model, data):
    inputs = data["inputs"]
    gts = data["gts"]
    logger = LoggerTensorBoard(args.argsCommon.experiment_save_path, args.argsCommon.experiment_save_path +'/tensorboard')
    # Tensorboard Adding graph test
    logger.addGraph(my_model, [inputs], verbose=False)

    # Tensorboard Adding image test

    irPair = torch.cat((inputs[0], gts[0]),0)
    visiblePair = torch.cat((inputs[1], gts[1]),0)
    irPair = torch.cat((irPair, irPair, irPair), 1)

    logger.addImageGrid(irPair, 'ir_image_pairs', 1)
    logger.addImageGrid(visiblePair, 'visible_image_pairs', 1)
    logger.addImageGrid(torch.cat((irPair, visiblePair), 0), 'combined_image_pair', 1, nrow=irPair.shape[0])

    # Using default templates test
    epochLogTxt = logger.getDefaultLogTemplates('epochLog', [1, 0.003])
    validationTxt = logger.getDefaultLogTemplates('validationLog', [400, 1200, 0.27, 18.3, 0.63])
    bestLoss = logger.getDefaultLogTemplates('bestValidationLoss', [128, 0.11])
    print(epochLogTxt)
    print(validationTxt)
    print(bestLoss)

    # Adding Scalar test
    for i in range(1000):
        logger.addScalar('loss', i*0.1, i)
        logger.addScalar('psnr', 10+i/40, i)

    # Log Image save to file test
    logger.logImage(inputs[1], 'input')
    logger.logImage(gts[1], 'result')

    logger.logImage(inputs[0], 'input_ir')
    logger.logImage(gts[0], 'result_ir')

    grid = torchvision.utils.make_grid(torch.cat((irPair, visiblePair), 0), nrow=irPair.shape[0]).unsqueeze(dim=0)
    visibleGrid = torchvision.utils.make_grid(visiblePair, nrow=irPair.shape[0]).unsqueeze(dim=0)
    logger.logImage(grid, 'grid')
    logger.logImage(visibleGrid, 'visibleGrid')

    # Log text save to file test
    i = 0
    logger.logText(logger.getDefaultLogTemplates('trainLoss', [i * 0.1]), 'loss', force_reset=True)
    logger.logText(logger.getDefaultLogTemplates('bestBenchmarkResult', [i + 1, 'PSNR', 10 + i / 40]), 'psnr', force_reset=True)
    for i in range(1, 1000):
        logger.logText(logger.getDefaultLogTemplates('trainLoss', [i*0.1]), 'loss')
        logger.logText(logger.getDefaultLogTemplates('bestBenchmarkResult', [i+1, 'PSNR', 10+i/40]), 'psnr')

if __name__ == '__main__':
    main()
