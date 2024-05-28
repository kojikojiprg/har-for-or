from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from webdataset import WebDataset


class DataModule(LightningDataModule):
    def __init__(self, dataset: WebDataset, batch_size: int):
        super().__init__()
        self.prepare_data_per_node = True
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        self.dataset = self.dataset.shuffle(3e9)
        return DataLoader(
            self.dataset, self.batch_size, num_workers=16, pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, self.batch_size, num_workers=16, pin_memory=True
        )
