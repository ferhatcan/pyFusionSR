import torch.nn as nn
import torch.optim as optim

from methods.baseMethod import BaseMethod

def getBaseMethod(model: nn.Module,  loss_functions: dict, optimizer: optim, args):
    return BaseMethod(model, loss_functions, optimizer, args)