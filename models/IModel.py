import torch.nn as nn

class IModel(nn.Module):
    def __init__(self):
        super(IModel, self).__init__()

    def forward(self, inputs: list):
        raise NotImplementedError