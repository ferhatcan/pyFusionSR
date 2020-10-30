import os
import glob
import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from PIL import Image


def checkNameValidty(name):
    assert name != '', 'name for logging a text should be valid. Desired Format is following: \n Not empty, not includes extension, not inludes point --> loss, psnr ...'
    assert len(name.split('.')) == 1, 'name can not include \'.\' or extention'

class Logger:
    templates = {
        'epochSummary': {'input_num': 1, 'template': "[Epoch {:4d}] Training results: \n"},
        'epochLog': {'input_num': 2, 'template': '[Epoch {}]\tLearning rate: {:.4f}\n'},
        'validationLog': {'input_num': 5, 'template': '[{}/{}]\t{}\t{:.1f}+{:.1f} msec\n'},
        'trainLoss': {'input_num': 2, 'template': '{} Train Loss: {:.4f}\n'},
        'benchmark': {'input_num': 2, 'template': '{} Benchmark Result: {:.4f}\n'},
        'validationLoss': {'input_num': 2, 'template': '{} Validation Loss: {:.4f}\n'},
        'bestValidationLoss': {'input_num': 2, 'template': 'Evaluation at epoch {} reached lowest Fusion validation set effective loss: {:.4f}\n'},
        'bestBenchmarkResult': {'input_num': 3, 'template': 'Evaluation at epoch {} reached best {} benchmark result: {:.3f}\n'}
    }

    def __init__(self, destination_folder):
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)
        print('Logger is activated. Destination for logs is --> [{}]'.format(destination_folder))
        self.destination_folder = destination_folder

        self.destination_logImage = os.path.join(self.destination_folder, 'image_logs')
        if not os.path.isdir(self.destination_logImage):
            os.mkdir(self.destination_logImage)

        self.destination_logText = os.path.join(self.destination_folder, 'text_logs')
        if not os.path.isdir(self.destination_logText):
            os.mkdir(self.destination_logText)

        self.destination_checkpoint = os.path.join(self.destination_folder, 'checkpoint')
        if not os.path.isdir(self.destination_checkpoint):
            os.mkdir(self.destination_checkpoint)

    def logImage(self, image: torch.tensor, fileName, overwrite=False):
        if fileName == '':
            fileName = 'image-' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        saveName = os.path.join(self.destination_logImage, fileName)
        if glob.glob(saveName + '*.*') and not overwrite:
            fileName = fileName + datetime.datetime.now().strftime("%H:%M:%S")
            saveName = os.path.join(self.destination_logImage, fileName)
        saveImage = image.cpu().detach()
        trans = torchvision.transforms.ToPILImage()
        for i in range(saveImage.shape[0]):
            im = saveImage[i, ...]
            convertedImage = trans(im)
            convertedImage.save(saveName + '_{}.png'.format(i))

    def logText(self, text, fileName, force_reset=False, verbose=False):
        checkNameValidty(fileName)

        savePath = os.path.join(self.destination_logText, fileName) + '.txt'
        if force_reset and os.path.exists(savePath):
            os.remove(savePath)
        with open(savePath, 'a') as file:
            file.write(text)
        if verbose:
            print(text, end='')

    def saveCheckpoint(self, stateDict, fileName):
        checkNameValidty(fileName)

        assert 'model_state_dict' in stateDict, 'checkpoint should include model parameters'
        assert 'optimizer' in stateDict, 'checkpoint should include optimizer key(it can be empty but key should exist)'
        assert 'epoch' in stateDict, 'checkpoint should include epoch key(it can be empty but key should exist)'

        savePath = os.path.join(self.destination_checkpoint, fileName) + '.pth'
        torch.save(stateDict, savePath)

    def loadCheckpoint(self, fileName):
        checkNameValidty(fileName)

        load_path = os.path.join(self.destination_checkpoint, fileName) + '.pth'
        assert os.path.exists(load_path), 'There is no state exist'

        return torch.load(load_path)

    @staticmethod
    def getDefaultLogTemplates(type_name, inputs: list):
        assert type_name in Logger.templates, 'There is no saved template for {} name'.format(type_name)
        assert len(inputs) == Logger.templates[type_name]['input_num'], 'There should be {} input(s) given {} input(s)'.format(Logger.templates[type_name]['input_num'], len(inputs))

        logTxt = Logger.templates[type_name]['template'].format(*inputs)
        return logTxt


class LoggerTensorBoard(Logger):
    def __init__(self, destination_folder, writerName):
        super(LoggerTensorBoard, self).__init__(destination_folder)

        print('Tensorboard Summary is saved to --> [{}]'.format(writerName))
        self.writer = SummaryWriter(writerName)

    def addImageGrid(self, batch_image: torch.tensor, tag, epoch_num, nrow=-1):
        """
        Add image to summary

        Args:
            batch_image: 4D torch.tensor
            tag: tag of the logging data saved to tensorboard
            epoch_num: global step value to record
            nrow: (int) row number in image grid
        """
        checkNameValidty(tag)
        if nrow <= 0:
            nrow = batch_image.shape[0]

        grid = torchvision.utils.make_grid(batch_image, nrow=nrow)
        self.writer.add_image(tag, grid, epoch_num)
        self.writer.close()

    def addGraph(self, model, input_to_model, verbose=False):
        """
        Add graph data to summary.

        Args:
            model (torch.nn.Module): Model to draw.
            input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
                variables to be fed.
            verbose (bool): Whether to print graph structure in console.
        """
        self.writer.add_graph(model, input_to_model, verbose)
        self.writer.close()

    def addScalar(self, tag, value, global_step):
        """
        Add scalar value to summary

        Args
            :param tag: (string) Data identifier
            :param value: (float or string) value to save
            :param global_step: (int) global step value to record
            :return: None
        """
        self.writer.add_scalar(tag, value, global_step)
        self.writer.close()