import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from methods.IMethod import IMethod


class BaseMethod(IMethod):
    def __init__(self, model: nn.Module,  loss_functions: dict, optimizer: optim, args):
        super(BaseMethod, self).__init__(model, loss_functions, optimizer, args)

        self.gclip = args.gclip if args.gclip > 0 else 0

    def train(self, data):
        self.model.train()
        # clear gradients
        self.optimizer.zero_grad()
        # run model
        result = self.model(data["inputs"])
        data["result"] = [result]
        # calculate loss
        losses = self._calculateLoss(data)
        self._checkNaN(losses)
        # calculate gradients
        losses[-1].backward()
        # do gradient clipping
        if self.gclip > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.gclip
            )
        # back-propagation
        self.optimizer.step()

        return result, losses

    def validate(self, data):
        self.model.eval()
        # run model
        with torch.no_grad():
            result = self.model(data["inputs"])
        data["result"] = [result]
        # calculate loss
        losses = self._calculateLoss(data)
        self._checkNaN(losses)
        
        return result, losses

    def test(self, data):
        self.model.eval()
        # run model
        with torch.no_grad():
            result = self.model(data["inputs"])

        return result

    def _calculateLoss(self, inputs):
        losses = []
        for weight, loss_function in zip(self.loss_functions['weights'], self.loss_functions['functions']):
            results = loss_function(inputs)
            losses.append(weight * sum(results) / len(results))
        losses.append(sum(losses))
        return losses

    def _checkNaN(self, inputs):
        for i, inp in enumerate(inputs):
            if np.isnan(inp.cpu().detach().numpy()):
                raise ValueError('Loss function returns NaN for loss type {}'.format(self.loss_functions['types'][i]))

