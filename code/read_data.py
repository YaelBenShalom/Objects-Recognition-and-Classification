from torch.utils.data.dataset import Dataset
from torchvision import datasets

from load_data import load_dataset

class ReadDataset(Dataset):
    def __init__(self, dataset_name, transform=None, max_items=None):
        self.features, self.labels = load_dataset(dataset_name)
        self.transform = transform
        self.max_items = max_items

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        if self.max_items:
            return self.max_items
        return len(self.labels)