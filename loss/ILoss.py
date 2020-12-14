import torch.nn as nn
import kornia as kn

class ILoss(nn.Module):
    def __init__(self):
        super(ILoss, self).__init__()

    def forward(self, data: dict) -> list:
        raise NotImplementedError

    @staticmethod
    def convert2singleChannel(image):
        if image.shape[1] == 1:
            return image
        elif image.shape[1] == 3:
            return kn.apply_grayscale(image)
        else:
            raise BaseException
