from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

DESIRED_ASSEMBLER = "fusionv2ADAS"
CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2ADAS_YsingleChannels.ini"

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER, CONFIG_FILE_NAME)
    experiment.load('model_last')
    experiment.train()

if __name__ == '__main__':
    main()

