"""
Author  : Xu fuyong
Time    : created by 2019/7/16 19:49

"""
import h5py
import numpy as np
from torch.utils.data import Dataset

from pathlib import Path
import os
import logging
from PIL import Image
import torch

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :]/255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


# label: HR image
# data: LR image
class DRealSRDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, scale: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.scale = scale

        self.ids = [os.splitext(file)[0] for file in os.listdir(label_dir)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        
        logging.info(f"Creating dataset with {len(self.ids)} examples")
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        label_name = self.ids[idx]
        data_name = label_name.replace("_x1_", f"_x{self.scale}_")

        label_file = list(self.label_dir.glob(label_name + '.*'))
        data_file = list(self.data_file.glob(data_name + '.*'))

        assert len(data_file) == 1, f'Either no data or multiple datas found for the ID {data_name}: {data_file}'
        assert len(label_file) == 1, f'Either no label or multiple label found for the ID {label_name}: {label_file}'

        data = Image.open(data_file[0])
        label = Image.open(label_file[0])
        
        assert data.size == label.size, \
            f'data and label {label_name} should be the same size, but are {data.size} and {label.size}'
        
        return {
            'data': torch.as_tensor(data.copy()).float().contiguous(),
            'label': torch.as_tensor(label.copy()).long().contiguous()
        }














