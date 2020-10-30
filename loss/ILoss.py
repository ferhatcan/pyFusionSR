import torch.nn as nn

class ILoss(nn.Module):
    def __init__(self):
        super(ILoss, self).__init__()

    def forward(self, data: dict) -> list:
        raise NotImplementedError