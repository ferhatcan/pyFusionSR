import os
import torch
import torch.nn as nn

from models.encoderdecoder.encoders import EncoderPyramid, EncoderPyramidv2
from models.encoderdecoder.decoders import DecoderPyramid, DecoderPyramidv2
from models.IModel import IModel


class EncoderDecoderFusion(IModel):
    def __init__(self, args):
        super(EncoderDecoderFusion, self).__init__()

        self.ir_model = EncoderDecoderPyramidv2(args, args.ir_channel_number)
        self.eo_model = EncoderDecoderPyramidv2(args, args.eo_channel_number)

        self.ir_model.load_state_dict(torch.load(args.ir_pretrained_weights))
        self.eo_model.load_state_dict(torch.load(args.eo_pretrained_weights))

        # for param in self.ir_model.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.eo_model.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, inputs: list):
        assert len(inputs) == 2, 'There should be 2 inputs'
        feats_ir = self.ir_model.encoder(inputs[0])
        feats_rgb = self.eo_model.encoder(inputs[1])

        # concatanete feats --> generate feats
        for key, _ in feats_ir.items():
            feats_ir[key] = (feats_ir[key] + feats_rgb[key]) / 2

        out = self.eo_model.decoder(feats_ir)

        return out


class EncoderDecoderPyramidv2(IModel):
    def __init__(self, args, channel_number=None):
        super(EncoderDecoderPyramidv2, self).__init__()

        if channel_number is None:
            channel_number = args.channel_number

        self.encoder = EncoderPyramidv2(args, input_dim=channel_number)
        self.decoder = DecoderPyramidv2(args, output_dim=channel_number, scale=args.scale)

    def forward(self, inputs: list):
        assert len(inputs) == 1, 'There should be 1 input'
        feats = self.encoder(inputs[0])
        out = self.decoder(feats)
        return out


class EncoderDecoderPyramid(IModel):
    def __init__(self, args):
        super(EncoderDecoderPyramid, self).__init__()

        self.encoder = EncoderPyramid(args, input_dim=args.channel_number)
        self.decoder = DecoderPyramid(args, output_dim=args.channel_number, scale=args.scale)

    def forward(self, inputs: list):
        assert len(inputs) == 1, 'There should be 1 input'
        feats = self.encoder(inputs[0])
        out = self.decoder(feats)
        return out


class EncoderFusionDecoderv2(IModel):
    def __init__(self, args):
        super(EncoderFusionDecoderv2, self).__init__()
        self.model_ir = EncoderDecoderPyramidv2(args=args, channel_number=args.ir_channel_number)
        self.model_eo = EncoderDecoderPyramidv2(args=args, channel_number=args.eo_channel_number)

        self.fusion_layers = [nn.Conv2d(2048, 1024, kernel_size=(1,1)),
                              nn.Conv2d(1024, 512, kernel_size=(1,1)),
                              nn.Conv2d(512, 256, kernel_size=(1,1)),
                              nn.Conv2d(256, 128, kernel_size=(1,1)),
                              nn.Conv2d(128, 64, kernel_size=(1,1)),]

        self.fusion_layers = nn.ModuleList(self.fusion_layers)

        self.model_ir.load_state_dict(torch.load(args.ir_pretrained_weights))
        self.model_eo.load_state_dict(torch.load(args.eo_pretrained_weights))

        self.out_layer = self.model_ir.decoder if args.output == "ir" else self.model_eo.decoder

    def forward(self, inputs: list):
        assert len(inputs) == 2, 'There should be 2 inputs'
        feat_ir = self.model_ir.encoder(inputs[0])
        feat_eo = self.model_eo.encoder(inputs[1])

        fusion = dict()
        for index, key in enumerate(feat_ir):
            currIndex = len(self.fusion_layers) - index - 1
            fusion[key] = self.fusion_layers[currIndex](torch.cat((feat_ir[key], feat_eo[key]), dim=1))

        output = self.out_layer(fusion)

        return output

    # def _apply(self, fn):
    #     super(EncoderFusionDecoderv2, self)._apply(fn)
    #     # self.fusion_layers = [fn(layer) for layer in self.fusion_layers]
    #     for i, layer in enumerate(self.fusion_layers):
    #         self.fusion_layers[i] = fn(self.fusion_layers[i])
    #     return self

# # testing puposes
#
# from dataloaders.div2k import div2K
# from dataloaders.irChallangeDataset import irChallangeDataset
# from options import options
# import matplotlib.pyplot as plt
#
#
# CONFIG_FILE_NAME = "./../../configs/encoderDecoder.ini"
# args = options(CONFIG_FILE_NAME)
#
# args.train_set_paths = ('/media/ferhatcan/New Volume/Image_Datasets/ir_sr_challange/train/640_flir_hr', )
# args.test_set_paths = ('/media/ferhatcan/New Volume/Image_Datasets/ir_sr_challange/test/640_flir_hr', )
#
# dl_div2k = div2K(args, train_path=('/media/ferhatcan/New Volume/Image_Datasets/div2k/images/train/DIV2K_train_HR', ),
#                  test_path= ('/media/ferhatcan/New Volume/Image_Datasets/div2k/images/validation/DIV2K_valid_HR', ))
#
# dl_ir = irChallangeDataset(args)
#
# # for batch, ((lr_ir, hr_ir), (lr_eo, hr_eo)) in enumerate(zip(dl_ir.loader_train, dl_div2k.loader_train)):
# #     temp = 0
#
# lr_eo, hr_eo = next(iter(dl_div2k.loader_train))
# lr_ir, hr_ir = next(iter(dl_ir.loader_train))
#
#
# # plt.ion()
# # image = hr_eo[0, ...].unsqueeze(dim=0)
# # im = image.permute(0, 2, 3, 1).squeeze()
# # plt.imshow(hr_eo[0, 0, ...].squeeze())
# # plt.waitforbuttonpress()
# #
# # plt.figure()
# # plt.imshow(im)
#
# model = EncoderDecoderPyramidv2(args=args)
# loss_function = nn.MSELoss()
#
# optim = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# model.to('cuda')
# hr_eo = hr_eo.to('cuda')
# lr_ir = lr_ir.to('cuda')
# hr_ir = hr_ir.to('cuda')
#
# model.train()
# sr_ir = model(lr_ir)
# tmp = 0
# loss_ir = loss_function(sr_ir, hr_ir)
# loss_eo = loss_function(eo, hr_eo)
# loss = loss_eo + loss_ir
# loss.backward()
# for p in model.encoder.encoder_shared.parameters():
#     p.grad.data = 0.5 * p.grad.data
# for p in model.decoder.decoder_shared.parameters():
#     p.grad.data = 0.5 * p.grad.data
#
# optim.step()
# model.zero_grad()
#
# torch.save(model.state_dict(), './../../.pre_trained_weights/encoderDecoder.pth')
#
# modelfusion = EncoderFusionDecoder(args=args, load_path='./../../.pre_trained_weights/encoderDecoder.pth')
# modelfusion.to('cuda')
# lr_eo = lr_eo.to('cuda')
# sr_ir = modelfusion(lr_ir, lr_eo)
#
# print(modelfusion)
# tmp = 0