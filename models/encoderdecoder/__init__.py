from .encoderDecoderSeperate import *

__all__ = ['EncoderFusionDecoderv2', 'EncoderDecoderPyramidv2', 'EncoderDecoderFusion']

def make_model(args):
    if args.type == 'seperatev2':
        return EncoderDecoderPyramidv2(args)
    elif args.type == 'fusion':
        return EncoderDecoderFusion(args)
    elif args.type == "fusionv2":
        return EncoderFusionDecoderv2(args)