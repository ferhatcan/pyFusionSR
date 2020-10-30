import torch
import torchvision
import torch.nn as nn

model = torchvision.models.resnet18()

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DecoderPyramidv2(nn.Module):
    def __init__(self, args, output_dim=1, scale=2):
        super(DecoderPyramidv2, self).__init__()
        self.args = args
        # self.device =  torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        assert (scale%2 == 0 or scale == 1), 'Scale should be multiple of 2 or be 1'

        self.layer1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)

        self.layers_scale = []
        for _ in range(scale//2):
            self.layers_scale.append(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False))
            #self.layer_scale[-1].to(self.device)
        if not len(self.layers_scale) == 0:
            self.layer_scale = nn.Sequential(*self.layers_scale)

        self.layer6 = nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.tanh = nn.Tanh()

        self.output_layer = nn.Sequential(
            self.layer6,
            self.tanh,
        )

    def forward(self, feats):
        assert len(feats) == 5, 'There should be 5 layer outputs for decoder!'

        out1 = self.layer1(feats['layer4'])
        out1 = torch.cat((feats['layer3'], out1), dim=1)

        out2 = self.layer2(out1)
        out2 = torch.cat((feats['layer2'], out2), dim=1)

        out3 = self.layer3(out2)
        out3 = torch.cat((feats['layer1'], out3), dim=1)

        out4 = self.layer4(out3)
        out4 = torch.cat((feats['input_layer'], out4), dim=1)

        out5 = self.layer5(out4)

        if not len(self.layers_scale) == 0:
            out5 = self.layer_scale(out5)

        out6 = self.output_layer(out5)

        return out6


class DecoderPyramid(nn.Module):
    def __init__(self, args, output_dim=1, scale=2):
        super(DecoderPyramid, self).__init__()
        self.args = args
        # self.device =  torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        assert (scale%2 == 0 or scale == 1), 'Scale should be multiple of 2 or be 1'

        self.layer1 = Bottleneck(2048, 1024, stride=2, groups=32, base_width=4)
        self.layer2 = Bottleneck(2048, 512, stride=2, groups=32, base_width=4)
        self.layer3 = Bottleneck(1024, 256, stride=2, groups=32, base_width=4)
        self.layer4 = Bottleneck(512, 64, stride=1, groups=32, base_width=4)
        self.layer5 = Bottleneck(128, 8, stride=2, groups=32, base_width=4)

        self.layers_scale = []
        for _ in range(scale//2):
            self.layers_scale.append(Bottleneck(8, 8, stride=2, groups=8, base_width=4))
            #self.layer_scale[-1].to(self.device)
        if not len(self.layers_scale) == 0:
            self.layer_scale = nn.Sequential(*self.layers_scale)

        self.layer6 = Bottleneck(8, 1, stride=2, groups=8, base_width=4) if output_dim == 1 \
                 else Bottleneck(8, 3, stride=2, groups=8, base_width=4)

        self.tanh = nn.Tanh()

        self.output_layer = nn.Sequential(
            self.layer6,
            self.tanh,
        )

    def forward(self, feats):
        assert len(feats) == 5, 'There should be 5 layer outputs for decoder!'

        out1 = self.layer1(feats['layer4'])
        out1 = torch.cat((feats['layer3'], out1), dim=1)

        out2 = self.layer2(out1)
        out2 = torch.cat((feats['layer2'], out2), dim=1)

        out3 = self.layer3(out2)
        out3 = torch.cat((feats['layer1'], out3), dim=1)

        out4 = self.layer4(out3)
        out4 = torch.cat((feats['input_layer'], out4), dim=1)

        out5 = self.layer5(out4)

        if not len(self.layers_scale) == 0:
            out5 = self.layer_scale(out5)

        out6 = self.output_layer(out5)

        return out6



class DecoderWeightShare(nn.Module):
    def __init__(self):
        super(DecoderWeightShare, self).__init__()

        self.layer1_ir = Bottleneck(2048, 512, stride=2, groups=32, base_width=4)
        self.layer2_ir = Bottleneck(512, 128, stride=2, groups=32, base_width=4)
        self.layer3_ir = Bottleneck(128, 32, stride=2, groups=32, base_width=4)
        self.layer3_2_ir = Bottleneck(32, 32, stride=2, groups=32, base_width=4)
        self.layer_5_ir = Bottleneck(32, 8, stride=2, groups=32, base_width=4)

        self.layer1_eo = Bottleneck(2048, 512, stride=2, groups=32, base_width=4)
        self.layer2_eo = Bottleneck(512, 128, stride=2, groups=32, base_width=4)
        self.layer3_eo = Bottleneck(128, 32, stride=2, groups=32, base_width=4)
        self.layer_5_eo = Bottleneck(32, 8, stride=2, groups=32, base_width=4)

        #self.layer_shared_4 = Bottleneck(32, 32, stride=1, groups=32, base_width=4)
        # self.layer_shared_5 = Bottleneck(32, 8, stride=2, groups=32, base_width=4)

        self.layer_ir_5 = Bottleneck(8, 1, stride=2, groups=8, base_width=4)
        self.layer_eo_5 = Bottleneck(8, 3, stride=2, groups=8, base_width=4)

        self.tanh = nn.Tanh()

        self.decoder_ir = nn.Sequential(
            self.layer1_ir,
            self.layer2_ir,
            self.layer3_ir,
            self.layer3_2_ir,
            self.layer_5_ir,
        )

        self.decoder_eo = nn.Sequential(
            self.layer1_eo,
            self.layer2_eo,
            self.layer3_eo,
            self.layer_5_eo,
        )

        # self.decoder_shared = nn.Sequential(
        #     #self.layer_shared_4,
        #     self.layer_shared_5,
        # )

    def forward(self, feat_ir, feat_eo):
        h_ir = self.decoder_ir(feat_ir)
        # h_ir = self.decoder_shared(h_ir)
        h_ir = self.tanh(self.layer_ir_5(h_ir))

        h_eo = self.decoder_eo(feat_eo)
        # h_eo = self.decoder_shared(h_eo)
        h_eo = self.tanh(self.layer_eo_5(h_eo))

        return h_ir, h_eo

    def forward_ir(self, feat_ir):
        h_ir = self.decoder_ir(feat_ir)
        # h_ir = self.decoder_shared(h_ir)
        h_ir = self.tanh(self.layer_ir_5(h_ir))
        return h_ir

    def forward_eo(self, feat_eo):
        h_eo = self.decoder_eo(feat_eo)
        # h_eo = self.decoder_shared(h_eo)
        h_eo = self.tanh(self.layer_eo_5(h_eo))
        return h_eo


class Bottleneck(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=(2+stride, 2+stride), stride=stride, padding=(1, 1),
                           groups=groups, bias=False)
        self.bn2 = norm_layer(inplanes)
        self.conv3 = conv1x1(inplanes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=(2 + stride, 2 + stride), stride=stride,
                                   padding=(1, 1), bias=False),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
