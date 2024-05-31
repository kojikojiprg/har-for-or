from lightning.pytorch import LightningDataModule
from webdataset import WebDataset, WebLoader


class DataModule(LightningDataModule):
    def __init__(self, dataset: WebDataset, num_workers: int = 8):
        super().__init__()
        self.prepare_data_per_node = True
        self.dataset = dataset
        self.num_workers = num_workers

    def train_dataloader(self):
        return WebLoader(self.dataset, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return WebLoader(self.dataset, num_workers=self.num_workers, pin_memory=True)
