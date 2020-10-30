import torch


class IDataLoader(torch.utils.data.Dataset):
    """
    Interface Class for data loading. Other than __init__ method should be
    overrided to make suitable interface.
    __getitem__: return a list of images not dictionary
    """
    def __init__(self):
        super(IDataLoader, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item: int) -> dict:
        raise NotImplementedError

    def transform(self, image):
        raise NotImplementedError

    @staticmethod
    def fillOutputDataDict(inputs: list, gts: list) -> dict:
        raise NotImplementedError