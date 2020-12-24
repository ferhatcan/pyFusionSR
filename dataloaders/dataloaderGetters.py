import torch
import numpy as np

from dataloaders.baseDataset import BaseDataset
from dataloaders.kaistDataset import KaistDataset
from dataloaders.flirAdasDataset import FlirAdasDataset


def _generateDataLoaders(ds_train, ds_test, batch_size, validation_percent=0.1, isShuffle=True) -> dict:
    dataset_size = len(ds_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_percent * dataset_size))
    if isShuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    valid_ds = torch.utils.data.Subset(ds_train, val_indices)
    train_ds = torch.utils.data.Subset(ds_train, train_indices)

    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    # validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    #
    # loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    # loader_val = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=validation_sampler, num_workers=0)
    # loader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=16)
    loader_val = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=16)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=8)

    loaders = dict()
    loaders["train"] = loader_train
    loaders["validation"] = loader_val
    loaders["test"] = loader_test

    return loaders


def getIrChallangeDataLoaders(args):
    ds_train = BaseDataset(args.train_set_paths, args)
    ds_test = BaseDataset(args.test_set_paths, args)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.validation_size, args.shuffle_dataset)


def getKaistDataLoaders(args):
    ds_train = KaistDataset(args=args, train=True)
    ds_test = KaistDataset(args=args, train=False)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.validation_size, args.shuffle_dataset)

def getFlirAdasDataLoaders(args):
    ds_train = FlirAdasDataset(args=args, train=True)
    ds_test = FlirAdasDataset(args=args, train=False)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.validation_size, args.shuffle_dataset)

def getFlirAdasKaistCombinedDataLoaders(args):
    train_set_paths = args.train_set_paths
    test_set_paths = args.test_set_paths
    args.train_set_paths = [train_set_paths[0]]
    args.test_set_paths = [test_set_paths[0]]
    ds_train_KAIST = KaistDataset(args=args, train=True)
    ds_test_KAIST = KaistDataset(args=args, train=False)
    args.train_set_paths = [train_set_paths[1]]
    args.test_set_paths = [test_set_paths[1]]
    ds_train_Flir = FlirAdasDataset(args=args, train=True)
    ds_test_Flir = FlirAdasDataset(args=args, train=False)

    ds_train = torch.utils.data.ConcatDataset([ds_train_KAIST, ds_train_Flir])
    ds_test = torch.utils.data.ConcatDataset([ds_test_KAIST, ds_test_Flir])

    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.validation_size, args.shuffle_dataset)

def getFlirAdasKaistMultipleDataLoaders(args):
    train_set_paths = args.train_set_paths
    test_set_paths = args.test_set_paths
    args.train_set_paths = [train_set_paths[0]]
    args.test_set_paths = [test_set_paths[0]]
    ds_train_KAIST = KaistDataset(args=args, train=True)
    ds_test_KAIST = KaistDataset(args=args, train=False)
    args.train_set_paths = [train_set_paths[1]]
    args.test_set_paths = [test_set_paths[1]]
    ds_train_Flir = FlirAdasDataset(args=args, train=True)
    ds_test_Flir = FlirAdasDataset(args=args, train=False)

    set1_dl = _generateDataLoaders(ds_train_KAIST, ds_test_KAIST, args.batch_size, args.validation_size, args.shuffle_dataset)
    set2_dl = _generateDataLoaders(ds_train_Flir, ds_test_Flir, args.batch_size, args.validation_size, args.shuffle_dataset)

    return [set1_dl, set2_dl]


