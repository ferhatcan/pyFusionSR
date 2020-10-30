class IExperiment:
    def __init__(self, args):
        self.args = args

    def train_one_epoch(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test_dataloader(self, dataloader):
        raise NotImplementedError

    def test_single(self, images: list):
        raise NotImplementedError

    def inference(self, data: dict):
        raise NotImplementedError

    def save(self, saveName: str):
        raise NotImplementedError

    def load(self, loadName: str):
        raise NotImplementedError