from keras.utils import Sequence
import h5py
import numpy as np
import random


class ReadSequence(Sequence):

    def __init__(self, file_path, batch_size):
        super().__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("train/data")
        self.label_x2 = hf.get("train/label_x2")
        self.label_x4 = hf.get("train/label_x4")
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil( self.data.shape[0] / float(self.batch_size) ))

    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.label_x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x4 = self.label_x4[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.stack([img
                            for img in batch_x])
        batch_x2 = np.stack([img
                            for img in batch_x2])
        batch_x4 = np.stack([img
                            for img in batch_x4])
        return batch_x, [batch_x2, batch_x4]