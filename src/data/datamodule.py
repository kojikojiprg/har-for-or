from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader
from webdataset import WebDataset, WebLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: WebDataset,
        dataset_type: str,
        batch_size: int,
        num_workers: int = 8,
    ):
        super().__init__()
        self.prepare_data_per_node = True
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        if self.dataset_type == "individual":
            return WebLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        elif self.dataset_type == "group":
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            raise ValueError

    def predict_dataloader(self):
        if self.dataset_type == "individual":
            return WebLoader(
                self.dataset,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        elif self.dataset_type == "group":
            return DataLoader(
                self.dataset,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            raise ValueError
