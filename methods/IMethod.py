import torch.nn as nn
import torch.optim as optim

class IMethod:
    """
    note: Inputs should be in same device (cpu or gpu)
    @train single training loop with ground thruth data to calculate loss
    @validate single test loop with ground thruth data to calculate loss
    @test single test loop without ground thruth data to calculate loss
    """
    def __init__(self, model: nn.Module, loss_functions: dict, optimizer: optim, args):
        self.model = model
        self.loss_functions = loss_functions
        self.optimizer = optimizer

    def train(self, data):
        raise NotImplementedError

    def validate(self, data):
        raise NotImplementedError

    def test(self, data):
        raise NotImplementedError