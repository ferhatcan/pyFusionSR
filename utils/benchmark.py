import torch
import numpy as np
from operator import lt, gt
import torch.nn as nn
from PIL import Image
from skimage.measure import *
from loss.fusionQualityLoss import FusionQualityEdgeLoss


class BenchmarkMetrics:
    validBenchmarkMethods = {
        'PSNR': {'function': compare_psnr, 'compare_func': gt, 'text': 'PSNR score is {:.2f}/{:.2f}', 'max_value': 100, 'start_value': 0, 'desired_input': 1, 'desired_ref': 1},
        'SSIM': {'function': compare_ssim, 'compare_func': gt, 'text':'SSIM score is {:.3f}/{:.2f}', 'max_value': 1, 'start_value': 0,  'desired_input': 1, 'desired_ref': 1},
        'MSE': {'function': compare_nrmse, 'compare_func': lt, 'text':'MSE score is {:.4f}. (Closer to {:.2f} is better)', 'max_value': 0, 'start_value': 1e8,  'desired_input': 1, 'desired_ref': 1},
        'L1': {'function': lambda x, y: np.mean(x - y), 'compare_func': lt, 'text':'L1 score is {:.4f}. (Closer to {:.2f} is better)', 'max_value': 0, 'start_value': 1e8,  'desired_input': 1, 'desired_ref': 1},
        'QE': {'function': None, 'compare_func': lt, 'text':'QE score is {:.4f}. (Closer to {:.2f} is better)', 'max_value': 0, 'start_value': 1e8,  'desired_input': 1, 'desired_ref': 2},
    }
    def __init__(self, args):
        self.args = args
        self.benchmark = dict()
        self.benchmark["method"] = []
        self.benchmark["name"] = []
        self.benchmark["compare_method"] = []
        for bench in args.benchmark_methods.split('+'):
            bench = bench.upper()
            if bench in BenchmarkMetrics.validBenchmarkMethods:
                self.benchmark["method"].append(getattr(self, 'get' + bench + 'score'))
                self.benchmark["name"].append(bench)
                self.benchmark["compare_method"].append(BenchmarkMetrics.validBenchmarkMethods[bench]["compare_func"])

    def getBenchmarkResults(self, data: dict):

        result = dict()
        for i, benchMethod in enumerate(self.benchmark["method"]):
            result[self.benchmark["name"][i]] = benchMethod(data)
            result[self.benchmark["name"][i]]["compare_method"] = self.benchmark["compare_method"][i]

        return result

    @staticmethod
    def checkConditions(data: dict, method: str):
        inputs = data["result"]
        gts = data["gts"]
        desired_ref = BenchmarkMetrics.validBenchmarkMethods[method]['desired_ref']
        desired_inp = BenchmarkMetrics.validBenchmarkMethods[method]['desired_input']
        assert len(inputs) >= desired_inp, 'There should be {:d} input(s) for calculating {} metric'.format(desired_inp, method)
        assert len(gts) >= desired_ref, 'There should be {:d} reference for calculating {} metric'.format(desired_ref, method)

        reference = gts[0]
        result = inputs[0]

        reference = reference.cpu().detach().numpy().transpose(0, 2, 3, 1)
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1)

        dataRange = 1
        normalize = False
        tmp = reference.max() - reference.min()
        if reference.max() - reference.min() <= 1:
            dataRange = 1
        elif reference.max() - reference.min() <= 255:
            dataRange = 255
        else:
            normalize = True

        if result.max() - result.min() > dataRange:
            normalize = True

        return normalize, dataRange

    @staticmethod
    def normalizeIf(image, isNorm):
        if isNorm:
            image -= image.min()
            image /= image.max()
        return image

    @staticmethod
    def calculateScores(reference, inputs, method: str, **kwargs) -> dict:
        scores = []
        scoreTxts = []

        func = BenchmarkMetrics.validBenchmarkMethods[method]['function']
        txt = BenchmarkMetrics.validBenchmarkMethods[method]['text']
        maxVal = BenchmarkMetrics.validBenchmarkMethods[method]['max_value']

        for i in range(reference.shape[0]):
            scores.append(func(reference[i, ...], inputs[i, ...], **kwargs))
            scoreTxts.append(txt.format(scores[-1], maxVal))

        return {'scores': scores, 'score_texts': scoreTxts}

    @staticmethod
    def getPSNRscore(inputs: dict) -> dict:
        normalize, dataRange = BenchmarkMetrics.checkConditions(inputs, 'PSNR')
        result = BenchmarkMetrics.normalizeIf(inputs['result'][0], normalize)
        for i in range(len(inputs['gts'])):
            if inputs['gts'][i].shape == result.shape:
                reference = BenchmarkMetrics.normalizeIf(inputs['gts'][i], normalize).cpu().detach().numpy().transpose(0, 2, 3, 1)
                break
        assert reference is not None, 'Input and reference shape must be matched.'
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1)
        return BenchmarkMetrics.calculateScores(reference, result, 'PSNR', data_range=dataRange)

    @staticmethod
    def getSSIMscore(inputs: dict):
        normalize, dataRange = BenchmarkMetrics.checkConditions(inputs, 'SSIM')
        result = BenchmarkMetrics.normalizeIf(inputs['result'][0], normalize)
        for i in range(len(inputs['gts'])):
            if inputs['gts'][i].shape == result.shape:
                reference = BenchmarkMetrics.normalizeIf(inputs['gts'][i], normalize).cpu().detach().numpy().transpose(0, 2, 3, 1)
                break
        assert reference is not None, 'Input and reference shape must be matched.'
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1)
        return BenchmarkMetrics.calculateScores(reference, result, 'SSIM', multichannel=True)

    @staticmethod
    def getMSEscore(inputs: dict):
        normalize, _ = BenchmarkMetrics.checkConditions(inputs, 'MSE')
        result = BenchmarkMetrics.normalizeIf(inputs['result'][0], normalize)
        for i in range(len(inputs['gts'])):
            if inputs['gts'][i].shape == result.shape:
                reference = BenchmarkMetrics.normalizeIf(inputs['gts'][i], normalize).cpu().detach().numpy().transpose(0, 2, 3, 1)
                break
        assert reference is not None, 'Input and reference shape must be matched.'
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1)
        return BenchmarkMetrics.calculateScores(reference, result, 'MSE')

    @staticmethod
    def getL1score(inputs: dict):
        normalize, _ = BenchmarkMetrics.checkConditions(inputs, 'L1')
        result = BenchmarkMetrics.normalizeIf(inputs['result'][0], normalize)
        for i in range(len(inputs['gts'])):
            if inputs['gts'][i].shape == result.shape:
                reference = BenchmarkMetrics.normalizeIf(inputs['gts'][i], normalize).cpu().detach().numpy().transpose(0, 2, 3, 1)
                break
        assert reference is not None, 'Input and reference shape must be matched.'
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1)
        return BenchmarkMetrics.calculateScores(reference, result, 'L1')

    def getQEscore(self, inputs: dict):
        # normalize, dataRange = BenchmarkMetrics.checkConditions(inputs, 'QE')

        scores = []
        scoreTxts = []

        txt = BenchmarkMetrics.validBenchmarkMethods["QE"]['text']
        maxVal = BenchmarkMetrics.validBenchmarkMethods["QE"]['max_value']

        args = self.args
        args.hr_shape = inputs["gts"][0].shape[-2:]
        lossQE = FusionQualityEdgeLoss(args)
        QEs = lossQE(inputs)
        for QE in QEs:
            scores.append(QE.item() * inputs["gts"][0].shape[0])
            scoreTxts.append(txt.format(scores[-1], maxVal))

        return {'scores': scores, 'score_texts': scoreTxts}

