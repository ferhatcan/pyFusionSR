import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def getLrScheduler(optimizer, milestones, gamma, last_epoch):
    return lrs.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)

def getADAMOptimizer(model_parameters, args):
    optimizer = optim.Adam(model_parameters, lr=args.learning_rate, betas=args.betas,
                      eps=args.epsilon, weight_decay=args.weight_decay)
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    lrScheduler = getLrScheduler(optimizer, milestones, args.decay_factor_gamma, -1)
    return lrScheduler, optimizer

def getSGDOptimizer(model_parameters, args):
    optimizer = optim.SGD(model_parameters, lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    lrScheduler = getLrScheduler(optimizer, milestones, args.decay_factor_gamma, -1)
    return lrScheduler, optimizer

