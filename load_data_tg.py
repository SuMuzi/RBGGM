from torch.utils.data.dataset import Dataset
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from scale import *
import scipy.ndimage


class TrainDataset(Dataset):
    def __init__(self, type, train_trmm_path, train_gpm_path, train_tem_path, train_hm_path, train_ref_path,
                 transform1=None,
                 transform2=None):
        super(TrainDataset, self).__init__()
        self.type = type
        self.trmm = os.listdir(train_trmm_path)
        self.gpm = os.listdir(train_gpm_path)
        self.tem = os.listdir(train_tem_path)
        self.hm = os.listdir(train_hm_path)
        self.ref = os.listdir(train_ref_path)

        self.transform1 = transform1
        self.transform2 = transform2
        self.h_path = 'D:/data/height.npy'

        self.trmm_and_gpm = []
        for i in range(len(self.gpm)):
            self.trmm_and_gpm.append(
                (
                    os.path.join(train_trmm_path + '/', self.trmm[i]),
                    os.path.join(train_gpm_path + '/', self.gpm[i]),
                    os.path.join(train_tem_path + '/', self.tem[i]),
                    os.path.join(train_hm_path + '/', self.hm[i]),
                    os.path.join(train_ref_path + '/', self.ref[i])
                )
            )
        self.border = [int(len(self.gpm) * 0), int(len(self.gpm) * 0.8), int(len(self.gpm) * 0.9), int(len(self.gpm))]
        print(self.border)

    def __getitem__(self, item):
        if self.type is 'train':
            start = self.border[0]

        if self.type is 'val':
            start = self.border[1]

        if self.type is 'test':
            start = self.border[2]

        trmm_path, gpm_path, tem_path, hm_path, ref_path = self.trmm_and_gpm[start + item]

        trmm_arr = np.load(trmm_path)
        trmm_arr = trmm_arr.T  # 经纬度变化
        trmm_arr = np.expand_dims(trmm_arr, 0)

        gpm_arr = np.load(gpm_path)
        gpm_arr = gpm_arr.T  # 经纬度变化
        # gpm_arr = np.log(gpm_arr + 1)
        gpm_arr = np.expand_dims(gpm_arr, 0)

        tem_arr = np.load(tem_path)
        tem_arr = tem_arr[2, :, :]
        tem_arr = np.expand_dims(tem_arr, axis=0)


        hm_arr = np.load(hm_path)
        hm_arr = hm_arr[2, :, :]
        hm_arr = np.expand_dims(hm_arr, axis=0)

        ref_arr = np.load(ref_path)
        ref_arr = ref_arr.T  # 经纬度变化
        ref_arr = np.expand_dims(ref_arr, 0)

        z = np.load(self.h_path)
        # z = scipy.ndimage.zoom(z, 0.625)
        z = np.expand_dims(z, 0)

        input_arr = np.concatenate((trmm_arr, tem_arr, hm_arr, z), axis=0)  # ,tem_arr, hm_arr
        # input_arr = trmm_arr
        if self.transform1 is not None:
            input_arr = torch.from_numpy(input_arr)
            input_arr = self.transform1(input_arr)

            gpm_arr = torch.from_numpy(gpm_arr)
            gpm_arr = self.transform2(gpm_arr)

        return np.array(input_arr).astype(np.float32), np.array(gpm_arr).astype(np.float32)  # .transpose(2, 0, 1)

    def __len__(self):
        if self.type is 'train':
            return int(len(self.gpm) * 0.8)

        if self.type is 'val':
            return int(len(self.gpm) * 0.1)

        if self.type is 'test':
            return int(len(self.gpm) * 0.1)
