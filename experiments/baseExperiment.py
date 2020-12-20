import torch
import gc
from operator import add

from experiments.IExperiment import IExperiment

class BaseExperiment(IExperiment):
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
        super(BaseExperiment, self).__init__(args)

        self.model = model
        self.dataloaders = dataLoaders
        self.loss = loss
        self.method = method
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.benchmark = benchmark
        self.logger = logger

        self.max_epoch = args.epoch_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        self.numOfLoss = len(self.loss['types'])
        self.numOfBenchmark = len(self.benchmark.benchmark["name"])
        # @todo save and load names of checkpoints ???
        # records holds all relevant information for recovering training
        self.records = dict()
        self.records["best_loss"] = 1e10
        self.records["best_validation_loss"] = {'values': [1e10] * self.numOfLoss, 'epoch_nums': [0] * self.numOfLoss}
        self.records["best_benchmarks"] = {'values': [0] * self.numOfBenchmark, 'epoch_nums': [0] * self.numOfBenchmark}
        for i, bench in enumerate(self.benchmark.benchmark["name"]):
            self.records["best_benchmarks"]["values"][i] = self.benchmark.validBenchmarkMethods[bench]["start_value"]
        self.records["model_state_dict"] = self.model.state_dict()
        self.records["optimizer"] = self.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.records["epoch"] = 0


    def train_one_epoch(self):
        total_losses = [0] * self.numOfLoss
        for batch_number, images in enumerate(self.dataloaders["train"]):
            for part in images:
                for i, image in enumerate(images[part]):
                    images[part][i] = image.to(self.device)
            _, losses = self.method.train(images)
            total_losses = list(map(add, total_losses, images["inputs"][0].shape[0] * losses))
            del images
            torch.cuda.empty_cache()
            if (batch_number + 1) % int(self.args.log_every * len(self.dataloaders["train"])) == 0:
                self.logger.logText(self.logger.getDefaultLogTemplates('validationLog',
                                                                       [(batch_number + 1) * self.args.batch_size,
                                                                        len(self.dataloaders["train"]) * self.args.batch_size,
                                                                        total_losses[-1] / int((batch_number + 1)), 0, 0]),
                                    'epochLog', verbose=True)
                if total_losses[-1] / int((batch_number + 1)) < self.records["best_loss"]:
                    self.records["best_loss"] = total_losses[-1] / int((batch_number + 1))

            if (batch_number + 1) % int(self.args.validate_every * len(self.dataloaders["train"])) == 0:
                validation_loss, validation_benchmark, benchmark_comparison_methods = self.test_dataloader(self.dataloaders["validation"])
                for i in range(self.numOfLoss):
                    if validation_loss[i] < self.records["best_validation_loss"]["values"][i]:
                        self.records["best_validation_loss"]["values"][i] = validation_loss[i]
                        self.records["best_validation_loss"]["epoch_nums"][i] = self.records["epoch"]
                        # @todo log best valid loss
                for i, bench in enumerate(validation_benchmark):
                    compare_method = benchmark_comparison_methods[i]
                    if compare_method(bench, self.records["best_benchmarks"]["values"][i]):
                        self.records["best_benchmarks"]["values"][i] = bench
                        self.records["best_benchmarks"]["epoch_nums"][i] = self.records["epoch"]
                        self.save("model_best")
                        # @todo add log best valid bench - also tensorboard

        self.lr_scheduler.step()
        average_losses = [x / (len(self.dataloaders["train"])) for x in total_losses]
        return average_losses

    def _tbImagelogs(self, data):
        # TensorBoard save samples lr-hr pairs
        inputs = data["inputs"].copy()
        gts = data["gts"].copy()
        result = data["results"].copy()

        for i, image in enumerate(inputs):
            if inputs[i].shape[1] == 1:
                inputs[i] = torch.cat((inputs[i], inputs[i], inputs[i]), 1)
            if gts[i].shape[1] == 1:
                gts[i] = torch.cat((gts[i], gts[i], gts[i]), 1)

        for i in range(len(result)):
            if result[i].shape[1] == 1:
                result[i] = torch.cat((result[i], result[i], result[i]), 1)

        irPair = torch.cat((inputs[0], gts[0]), 0)
        visiblePair = torch.cat((inputs[1], gts[1]), 0)
        resultPair = torch.cat((*inputs, *result), 0)

        self.logger.addImageGrid(irPair, 'ir_image_pairs', self.records["epoch"])
        self.logger.addImageGrid(visiblePair, 'visible_image_pairs', self.records["epoch"])
        self.logger.addImageGrid(torch.cat((irPair, visiblePair), 0), 'combined_image_pair', self.records["epoch"], nrow=irPair.shape[0])
        self.logger.addImageGrid(resultPair, 'input-result_image_pair', self.records["epoch"], nrow=irPair.shape[0])

    def _tbScalarLogs(self, training_loss, validation_loss, validation_benchmark):
        for i, loss_name in enumerate(self.loss['types']):
            self.logger.addScalar("training_loss_" + loss_name, training_loss[i], self.records["epoch"])
            self.logger.addScalar("validation_loss_" + loss_name, validation_loss[i], self.records["epoch"])
        for i, bench_name in enumerate(self.benchmark.benchmark["name"]):
            self.logger.addScalar("validation_benchmark_" + bench_name, validation_benchmark[i], self.records["epoch"])

    def _txtLogs(self, fileName, training_loss, validation_loss, validation_benchmark):
        self.logger.logText('----------------------------\nTraining Results...\n', fileName, verbose=True)
        for i, loss_name in enumerate(self.loss['types']):
            trainTxt = self.logger.getDefaultLogTemplates('trainLoss', [loss_name, training_loss[i]])
            valTxt = self.logger.getDefaultLogTemplates('validationLoss', [loss_name, validation_loss[i]])
            self.logger.logText(trainTxt, fileName, verbose=True)
            self.logger.logText(valTxt, fileName, verbose=True)
        for i, bench_name in enumerate(self.benchmark.benchmark["name"]):
            benchTxt = self.logger.getDefaultLogTemplates('benchmark', [bench_name, validation_benchmark[i]])
            self.logger.logText(benchTxt, fileName, verbose=True)

    def train(self):
        # @todo add timer
        referenceData = next(iter(self.dataloaders["validation"]))
        for part in referenceData:
            for i, image in enumerate(referenceData[part]):
                referenceData[part][i] = image.to(self.device)
        self.logger.addGraph(self.model, [referenceData["inputs"]], verbose=False)
        test_dict = self.test_single(referenceData)
        referenceData["results"]  = test_dict["results"]
        self._tbImagelogs(referenceData)
        print("Training Starts....")

        for i in range(self.records["epoch"], self.max_epoch):
            self.logger.logText(self.logger.getDefaultLogTemplates('epochLog',
                                                                   [self.records["epoch"] + 1,
                                                                    self.optimizer.param_groups[0]['lr']]),
                                'training_log', verbose=True)
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

    def test_dataloader(self, dataloader):
        total_losses = [0] * self.numOfLoss
        total_benchmark = [0] * self.numOfBenchmark
        benchmark_comparisons = []
        batch_num = 1
        for batch_number, images in enumerate(dataloader):
            benchmark_comparisons = []
            batch_num = images['inputs'][0].shape[0]
            test_dict = self.test_single(images)
            total_losses = list(map(add, total_losses, batch_num * test_dict["losses"]))
            images["result"] = test_dict["results"]
            benchmarkResults = self.benchmark.getBenchmarkResults(images)
            for index, bench in enumerate(benchmarkResults):
                total_benchmark[index] = total_benchmark[index] + sum(benchmarkResults[bench]['scores'])
                benchmark_comparisons.append(benchmarkResults[bench]["compare_method"])
            del images
            torch.cuda.empty_cache()
        average_losses = [x / (len(dataloader)) for x in total_losses]
        average_benchmark = [x / (len(dataloader) * self.args.batch_size) for x in total_benchmark]

        return average_losses, average_benchmark, benchmark_comparisons

    def test_single(self, data: dict):
        method_input = data.copy()
        output_dict = dict()
        output_dict["results"] = []
        output_dict["losses"] = []
        output_dict["bechmarks"] = []
        for i, image in enumerate(method_input["inputs"]):
            method_input["inputs"][i] = method_input["inputs"][i].to(self.device)
        for i, image in enumerate(method_input["gts"]):
            method_input["gts"][i] = method_input["gts"][i].to(self.device)
        result, losses  = self.method.validate(method_input)
        method_input["result"] = [result]
        benchmarkResults = self.benchmark.getBenchmarkResults(method_input)
        output_dict["results"] = [result]
        output_dict["losses"] = losses
        output_dict["bechmarks"] = benchmarkResults
        del method_input
        del result
        del losses
        torch.cuda.empty_cache()

        return output_dict

    def inference(self, data: dict):
        for i, imageList in enumerate(data["inputs"]):
            data["inputs"][i] = data["inputs"][i].to(self.device)
        result = self.method.test(data)
        data["results"] = [result]

        return data

    def save(self, saveName: str):
        self.records["model_state_dict"] = self.model.state_dict()
        self.records["optimizer"] = self.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.logger.saveCheckpoint(self.records, saveName)

    def load(self, loadName: str):
        # @todo there can be more than 1 new loss function. Not add only one value
        # @todo added value should be added with type of itself (greater or less)
        loaded_record = self.logger.loadCheckpoint(loadName)
        if loaded_record is None:
            return
        else:
            self.records = loaded_record
            if self.numOfBenchmark > len(self.records["best_benchmarks"]['values']):
                self.records["best_benchmarks"]["values"].append(0)
                self.records["best_benchmarks"]["epoch_nums"].append(0)
            if self.numOfLoss > len(self.records["best_validation_loss"]['values']):
                self.records["best_validation_loss"]["values"].append(0)
                self.records["best_validation_loss"]["epoch_nums"].append(0)
            self.model.load_state_dict(self.records["model_state_dict"])
            self.optimizer.load_state_dict(self.records["optimizer"])
            self.lr_scheduler.load_state_dict(self.records["lr_scheduler"])