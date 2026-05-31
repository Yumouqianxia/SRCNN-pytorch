import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self._h5 = None
        with h5py.File(self.h5_file, 'r') as f:
            self._length = len(f['lr'])

    @staticmethod
    def _to_chw(img):
        if img.ndim == 2:
            return np.expand_dims(img, 0)
        return np.transpose(img, (2, 0, 1))

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        lr = self._h5['lr'][idx].astype(np.float32) / 255.0
        hr = self._h5['hr'][idx].astype(np.float32) / 255.0
        return self._to_chw(lr), self._to_chw(hr)

    def __len__(self):
        return self._length

    def __del__(self):
        if hasattr(self, '_h5') and self._h5 is not None:
            self._h5.close()
            self._h5 = None


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self._h5 = None
        with h5py.File(self.h5_file, 'r') as f:
            self._keys = sorted(f['lr'].keys(), key=lambda x: int(x))
        self._length = len(self._keys)

    @staticmethod
    def _to_chw(img):
        if img.ndim == 2:
            return np.expand_dims(img, 0)
        return np.transpose(img, (2, 0, 1))

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        key = self._keys[idx]
        lr = self._h5['lr'][key][:].astype(np.float32) / 255.0
        hr = self._h5['hr'][key][:].astype(np.float32) / 255.0
        return self._to_chw(lr), self._to_chw(hr)

    def __len__(self):
        return self._length

    def __del__(self):
        if hasattr(self, '_h5') and self._h5 is not None:
            self._h5.close()
            self._h5 = None
