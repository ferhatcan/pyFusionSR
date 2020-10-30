from options import options
from utils.benchmark import BenchmarkMetrics

from dataloaders.dataloaderGetters import *
import models.encoderdecoder as model
from utils.logger import LoggerTensorBoard
from loss.lossGetters import *
from methods.methodGetters import *
from experiments.experimentGetters import *
from optimizers.optimizerSchedulerGetters import *

def getExperiment():
    CONFIG_FILE_NAME = "./configs/encoderDecoderFusionv2.ini"

    args = options(CONFIG_FILE_NAME)
    print("The system will use following resource: {:}".format(args.argsCommon.device))
    print("Experiment Name: " + args.argsCommon.experiment_name)
    print("Experiment will be saved to " + args.argsCommon.experiment_save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")

    dataloaders = getKaistDataLoaders(args.argsDataset)

    currentModel = model.make_model(args.argsModel)
    currentModel.to(device)

    possibles = globals().copy()
    loss_dict = dict()
    loss_dict["types"] = []
    loss_dict["functions"] = []
    loss_dict["weights"] = []
    for loss in args.argsLoss.loss.split('+'):
        weight, loss_type = loss.split('*')
        loss_dict["functions"].append(possibles.get('get'+loss_type+'Loss')(args.argsLoss))
        loss_dict["weights"].append(float(weight))
        loss_dict["types"].append(loss_type)

    lr_scheduler, optimizer = possibles.get('get'+args.argsMethod.optimizer+'Optimizer')(currentModel.parameters(), args.argsMethod)

    method = getBaseMethod(currentModel, loss_dict, optimizer, args.argsMethod)

    benchmark = BenchmarkMetrics(args.argsBenchmark)

    logger = LoggerTensorBoard(args.argsCommon.experiment_save_path,
                               args.argsCommon.experiment_save_path + '/tensorboard')

    experiment = getBaseExperiment(currentModel, dataloaders, loss_dict, method, optimizer, lr_scheduler, benchmark, logger, args.argsExperiment)

    return experiment