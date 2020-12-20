from assemblers.encoderDecoderFusionv2Assembler import getExperiment as fusionv2
from assemblers.encoderDecoderFusionv2ADASAssembler import getExperiment as fusionv2adas
from assemblers.encoderDecoderFusionv2ADASKAISTAssembler import getExperiment as fusionadaskaist

def getExperimentWithDesiredAssembler(name, config_file):
    if name == "fusionv2":
        return fusionv2(config_file)
    elif name == "fusionv2ADAS":
        return fusionv2adas(config_file)
    elif name == "fusionv2ADASKAIST":
        return fusionadaskaist(config_file)
    else:
        assert 1 == 0, "there should be invalid assembler name"
