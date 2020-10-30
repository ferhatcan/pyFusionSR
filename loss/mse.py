import torch.nn as nn

from loss.ILoss import ILoss

class MSELossLocal(ILoss):
    def __init__(self, args):
        super(MSELossLocal, self).__init__()

        self.loss_function = nn.MSELoss()

    def forward(self, data: dict) -> list:
        assert "gts" in data and "result" in data, "gts Type should be a dict and contains \"gts\" and \"result\" keys"
        assert len(data["result"]) == 1, "there should be 1 result to calculate loss"
        result = []
        for i in range(len(data["gts"])):
            if data["gts"][i].shape == data["result"][0].shape:
                result.append(self.loss_function(data["gts"][i], data["result"][0]))
        return result