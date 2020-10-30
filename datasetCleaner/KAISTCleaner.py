from dataloaders.kaistDataset import KaistDataset
from options import options
import torch

CONFIG_FILE_NAME = "../configs/encoderDecoderFusionv2.ini"
args = options(CONFIG_FILE_NAME)

dataset_train = KaistDataset(args.argsDataset, train=True)
dataset_test = KaistDataset(args.argsDataset, train=False)

def checkImageValidity(image):
    result1 = torch.isnan(image).any()
    result2 = torch.isinf(image).any()
    result3 = (image < 0).any()

    result = result1.__or__(result2)
    result = result.__or__(result3)
    return result

bad_indexs_inp = []
bad_indexs_gt = []
f = open('bad_sections.txt', 'a')
print(len(dataset_train))
for index in range(len(dataset_train)):
    if index % 1000 == 0:
        print(f'{index} file processed')
    data = dataset_train.__getitem__(index)
    for image in data['inputs']:
        if checkImageValidity(image):
            bad_indexs_inp.append(dataset_train.imageFiles[0][index])
            print(f'Invalid image path in inputs: {dataset_train.imageFiles[0][index]}, {index}')
            f.write(f'Invalid image path in inputs: {dataset_train.imageFiles[0][index]}, {index}')
    for image in data['gts']:
        if checkImageValidity(image):
            bad_indexs_gt.append(dataset_train.imageFiles[0][index])
            print(f'Invalid image path in gts: {dataset_train.imageFiles[0][index]},  {index}')
            f.write(f'Invalid image path in gts: {dataset_train.imageFiles[0][index]}, {index}')
