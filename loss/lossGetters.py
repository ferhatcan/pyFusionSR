from loss.fusionQualityLoss import FusionQualityEdgeLoss, FusionQualityLoss
from loss.mse import MSELossLocal
from loss.l1 import L1LossLocal

def getQLoss(args):
    return FusionQualityLoss(args)

def getQELoss(args):
    return FusionQualityEdgeLoss(args)

def getMSELoss(args):
    return MSELossLocal(args)

def getL1Loss(args):
    return L1LossLocal(args)