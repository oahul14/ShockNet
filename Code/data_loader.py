import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import os
from os.path import join as osjoin
import PIL.Image as Image

class ShockDataset(Dataset):
    def __init__(self, data_dir, transform_in=None, transform_out=None):
        """Parse the dataset into pytorch Dataset class

        Args:
            data_dir (str): directory where data were saved
        """
        self.in_dir = osjoin(data_dir, "InitialCondition")
        self.out_dir = osjoin(data_dir, "ResultsPNG")
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __len__(self):
        return len(os.listdir(self.out_dir))

    def __getitem__(self, idx):
        """get each tensor on the fly

        Args:
            idx (str): index of the dataset tensor

        Returns:
            sample_in (Tensor): sample input
            sample_out (Tensor): sample output
            fname (str): sample name
        """
        fname = os.listdir(self.in_dir)[idx]
        sample_in = np.asarray(Image.open(osjoin(self.in_dir, fname))).copy()
        sample_out = np.asarray(Image.open(osjoin(self.out_dir, fname))).copy()
        if self.transform_in:
            sample_in = self.transform_in(sample_in)
        else:
            sample_in = sample_in.reshape((3, sample_in.shape[0], -1))
        if self.transform_out:
            sample_out = self.transform_out(sample_out)
        else:
            sample_out = sample_out.reshape((3, sample_out.shape[0], -1))
        return sample_in, sample_out, fname

class Subset2Dataset(Dataset):
    def __init__(self, subset, transform=None):
        """Parse the Subset class into Dataset class for random splitting

        Args:
            subset (Subset): Subset to be parsed
        """
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y, fname = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y, fname
        
    def __len__(self):
        return len(self.subset)

