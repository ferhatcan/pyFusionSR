from assemblers.encoderDecoderFusionv2Assembler import getExperiment as fusionv2
from assemblers.encoderDecoderFusionv2ADASAssembler import getExperiment as fusionv2adas

def getExperimentWithDesiredAssembler(name):
    if name == "fusionv2":
        return fusionv2()
    elif name == "fusionv2ADAS":
        return fusionv2adas()
    else:
        assert 1 == 0, "there should be invalid assembler name"
