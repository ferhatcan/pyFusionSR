import torch.nn as nn

from loss.ILoss import ILoss

class L1LossLocal(ILoss):
    def __init__(self, args):
        super(L1LossLocal, self).__init__()

        self.loss_function = nn.L1Loss()

    def forward(self, data: dict) -> list:
        assert "gts" in data and "result" in data, "inputs Type should be a dict and contains \"inputs\" and \"result\" keys"
        assert len(data["result"]) == 1, "there should be 1 result to calculate loss"
        result = []
        for i in range(len(data["gts"])):
            if data["gts"][i].shape == data["result"][0].shape:
                result.append(self.loss_function(data["gts"][i], data["result"][0]))
        return result