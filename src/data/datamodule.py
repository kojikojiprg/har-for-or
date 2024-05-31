from lightning.pytorch import LightningDataModule
from webdataset import WebDataset, WebLoader


class DataModule(LightningDataModule):
    def __init__(self, dataset: WebDataset, batch_size: int):
        super().__init__()
        self.prepare_data_per_node = True
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return WebLoader(self.dataset, num_workers=2, pin_memory=True)

    def predict_dataloader(self):
        return WebLoader(self.dataset, self.batch_size, num_workers=2, pin_memory=True)
