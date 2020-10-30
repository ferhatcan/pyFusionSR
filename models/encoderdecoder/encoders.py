import torch
import copy
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EncoderPyramidv2(nn.Module):
    # This module takes an image as input (min. dimension should be 32x32)
    # There are 5 layers:
    #   - In each layer, input is downsampled
    #   - output of each layer is returned. (a dictionary that includes layer outputs)
    def __init__(self, args, input_dim=1):
        super(EncoderPyramidv2, self).__init__()
        self.args = args

        self.encoder_layer_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool = nn.AvgPool2d(2)

        self.input_layer = nn.Sequential(
            self.conv1,
            self.maxpool,
        )

    def forward(self, x):
        output = dict()
        output['input_layer'] = self.input_layer(x)
        output['layer1'] = self.maxpool(self.encoder_layer_1(output['input_layer']))
        output['layer2'] = self.maxpool(self.encoder_layer_2(output['layer1']))
        output['layer3'] = self.maxpool(self.encoder_layer_3(output['layer2']))
        output['layer4'] = self.maxpool(self.encoder_layer_4(output['layer3']))

        return output


class EncoderPyramid(nn.Module):
    # This module takes an image as input (min. dimension should be 32x32)
    # There are 5 layers:
    #   - In each layer, input is downsampled
    #   - output of each layer is returned. (a dictionary that includes layer outputs)
    def __init__(self, args, input_dim=1):
        super(EncoderPyramid, self).__init__()
        model = torch.hub.load(loadName)
        self.args = args

        self.encoder_layer_1 = copy.deepcopy(model.layer1[0])
        self.encoder_layer_2 = copy.deepcopy(model.layer2[0])
        self.encoder_layer_3 = copy.deepcopy(model.layer3[0])
        self.encoder_layer_4 = copy.deepcopy(model.layer4[0])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) if input_dim == 1 \
                    else copy.deepcopy(model.conv1)


        self.input_layer = nn.Sequential(
            self.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

    def forward(self, x):
        output = dict()
        output['input_layer'] = self.input_layer(x)
        output['layer1'] = self.encoder_layer_1(output['input_layer'])
        output['layer2'] = self.encoder_layer_2(output['layer1'])
        output['layer3'] = self.encoder_layer_3(output['layer2'])
        output['layer4'] = self.encoder_layer_4(output['layer3'])

        return output


class EncoderWeightShare(nn.Module):
    def __init__(self):
        super(EncoderWeightShare, self).__init__()
        model = torch.hub.load(loadName)

        self.encoder_layer_1_eo = copy.deepcopy(model.layer1[0])
        self.encoder_layer_2_eo = copy.deepcopy(model.layer2[0])
        self.encoder_layer_3_eo = copy.deepcopy(model.layer3[0])
        self.encoder_layer_4_eo = copy.deepcopy(model.layer4[0])

        self.encoder_layer_1_ir = copy.deepcopy(model.layer1[0])
        self.encoder_layer_2_ir = copy.deepcopy(model.layer2[0])
        self.encoder_layer_3_ir = copy.deepcopy(model.layer3[0])
        self.encoder_layer_4_ir = copy.deepcopy(model.layer4[0])

        self.conv1_eo = copy.deepcopy(model.conv1)
        self.conv1_ir = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.encoder_shared = nn.Sequential(
            model.bn1,
            model.relu,
            model.maxpool,
            # self.encoder_layer_shared,
        )
        self.encoder_eo = nn.Sequential(
            self.encoder_layer_1_eo,
            self.encoder_layer_2_eo,
            self.encoder_layer_3_eo,
            self.encoder_layer_4_eo
        )

        self.encoder_ir = nn.Sequential(
            self.encoder_layer_1_ir,
            self.encoder_layer_2_ir,
            self.encoder_layer_3_ir,
            self.encoder_layer_4_ir
        )

    def forward(self, image_ir, image_eo):
        h_ir = self.conv1_ir(image_ir)
        h_eo = self.conv1_eo(image_eo)

        h_ir = self.encoder_shared(h_ir)
        h_ir = self.encoder_ir(h_ir)

        h_eo = self.encoder_shared(h_eo)
        h_eo = self.encoder_eo(h_eo)

        return h_ir, h_eo

    def forward_ir(self, image_ir):
        h_ir = self.conv1_ir(image_ir)
        h_ir = self.encoder_shared(h_ir)
        h_ir = self.encoder_ir(h_ir)
        return h_ir

    def forward_eo(self, image_eo):
        h_eo = self.conv1_eo(image_eo)
        h_eo = self.encoder_shared(h_eo)
        h_eo = self.encoder_eo(h_eo)
        return h_eo
