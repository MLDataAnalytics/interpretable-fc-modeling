import os
import os.path

import pandas as pd
import numpy as np

import scipy.io as sio

import torch
from torch.utils.data import Dataset



class fcDataset(Dataset):
    def __init__(self, dat_file, dat_dir, fea_type='fc_p2p', label_idx=1, dat_xfm=None):
        self.dat_labels = pd.read_csv(dat_file, header=None)
        self.dat_dir = dat_dir
        self.fea_type = fea_type
        self.label_idx = label_idx
        self.dat_xfm = dat_xfm

    def __len__(self):
        return len(self.dat_labels)

    def __getitem__(self, idx):
        data_path = os.path.join(self.dat_dir, self.dat_labels.iloc[idx, 0])
        
        data = sio.loadmat(data_path)[self.fea_type]
        label = self.dat_labels.iloc[idx, self.label_idx]
        
        data[np.isnan(data)] = 0.0
        if self.dat_xfm == "fisherz":
            data = np.arctanh(data)
            data[np.isinf(data)] = 0.0

        return data, label
    
    def get_batch(self, batch_sz=1):
        num_dat = self.dat_labels.shape[0]
        choice_batch = np.random.choice(num_dat, batch_sz, False)
        
        data = []
        label = []
        for i in choice_batch:
            i_data_path = os.path.join(self.dat_dir, self.dat_labels.iloc[i, 0])
            i_data = sio.loadmat(i_data_path)[self.fea_type]
            i_label = self.dat_labels.iloc[i, self.label_idx]
            
            i_data = np.expand_dims(i_data, 0)
            i_label = np.expand_dims(i_label, 0)

            i_data[np.isnan(i_data)] = 0.0
            if self.dat_xfm == "fisherz":
                i_data = np.arctanh(i_data)
                i_data[np.isinf(i_data)] = 0.0

            data.append(i_data)
            label.append(i_label)

        data = torch.from_numpy(np.concatenate(data, 0))
        label = torch.from_numpy(np.concatenate(label, 0))
        
        return data, label

