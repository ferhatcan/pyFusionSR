import torch
import gc
from operator import add

from pympler import muppy, summary
from pympler import tracker
import objgraph

from experiments.baseExperiment import BaseExperiment

class MultipleDataloaderExperiment(BaseExperiment):
    def __init__(self,
                 model,
                 dataLoaders,
                 loss,
                 method,
                 optimizer,
                 lr_scheduler,
                 benchmark,
                 logger,
                 args):
        super(MultipleDataloaderExperiment, self).__init__(model, dataLoaders,
                                                             loss, method,
                                                             optimizer, lr_scheduler,
                                                             benchmark, logger, args)

        self.dataloaders_list = dataLoaders
        self.dataloaders = self.dataloaders_list[0] # initial dataloader

    def train(self):
        # @todo add timer
        referenceData = next(iter(self.dataloaders["validation"]))
        for part in referenceData:
            for i, image in enumerate(referenceData[part]):
                referenceData[part][i] = image.to(self.device)
        self.logger.addGraph(self.model, [referenceData["inputs"]], verbose=False)
        test_dict = self.test_single(referenceData)
        referenceData["results"] = test_dict["results"]
        self._tbImagelogs(referenceData)
        print("Training Starts....")

        for i in range(self.records["epoch"], self.max_epoch):
            self.logger.logText(self.logger.getDefaultLogTemplates('epochLog',
                                                                   [self.records["epoch"] + 1,
                                                                    self.optimizer.param_groups[0]['lr']]),
                                'training_log', verbose=True)
            for dl in self.dataloaders_list:
                print("DATALOADER CHANGE")
                self.dataloaders = dl
                average_losses = self.train_one_epoch()
                validation_loss, validation_benchmark, _ = self.test_dataloader(
                    self.dataloaders["validation"])
            self.records["epoch"] = self.records["epoch"] + 1
            self.save("model_last")
            test_dict = self.test_single(referenceData)
            referenceData["results"] = test_dict["results"]
            self._tbImagelogs(referenceData)
            self._tbScalarLogs(average_losses, validation_loss, validation_benchmark)
            self._txtLogs('epochLog', average_losses, validation_loss, validation_benchmark)