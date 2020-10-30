from experiments.baseExperiment import BaseExperiment

def getBaseExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args):
    return BaseExperiment(model, dataLoaders, loss, method, optimizer, lr_scheduler, benchmark, logger, args)