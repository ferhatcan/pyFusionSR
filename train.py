from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

DESIRED_ASSEMBLER = "fusionv2ADAS"

def main():
    experiment = getExperimentWithDesiredAssembler(DESIRED_ASSEMBLER)
    # experiment.load('model_last')
    experiment.train()

if __name__ == '__main__':
    main()

