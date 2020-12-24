from experiments.baseExperiment import BaseExperiment
from experiments.multipleDataloaderExperiment import MultipleDataloaderExperiment

def getBaseExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args):
    return BaseExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args)

def getMultipleDataloaderExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args):
    return MultipleDataloaderExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args)