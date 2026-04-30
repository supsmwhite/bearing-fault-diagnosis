"""Build PyTorch Dataset objects from processed CWRU npz files."""

import numpy as np
import torch
from torch.utils.data import Dataset


class CWRUWindowDataset(Dataset):
    def __init__(self, npz_path: str, indices=None, return_meta: bool = False):
        data = np.load(npz_path, allow_pickle=False)

        self.X = data["X"]
        self.y = data["y"]
        self.load_hp = data["load_hp"]
        self.class_name = data["class_name"]
        self.file_name = data["file_name"]
        self.start_index = data["start_index"]
        self.return_meta = return_meta

        if indices is None:
            self.indices = np.arange(len(self.y), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = self.indices[index]

        x = torch.as_tensor(self.X[sample_index], dtype=torch.float32).unsqueeze(0)
        y = torch.as_tensor(self.y[sample_index], dtype=torch.long)

        if not self.return_meta:
            return x, y

        meta = {
            "load_hp": int(self.load_hp[sample_index]),
            "class_name": str(self.class_name[sample_index]),
            "file_name": str(self.file_name[sample_index]),
            "start_index": int(self.start_index[sample_index]),
        }
        return x, y, meta

