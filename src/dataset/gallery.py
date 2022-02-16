import pickle, logging, numpy as np
from torch.utils.data import Dataset
import torch
import os
import glob
import random

class PoseDataset(Dataset):
    def __init__(self, root, 
                       inputs, 
                       num_frame, 
                       connect_joint, 
                       transform=None,
                       is_train=False):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        self.items_path = sorted(glob.glob(os.path.join(root, "*/*.npy"), recursive=True))
        self.labels = {key: value for value, key in enumerate(os.listdir(root))}
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.items_path)

    def _get_data(self, data):
        data = data.transpose(2,0,1)         
        data = np.expand_dims(data, axis=3)
        joint, velocity, bone = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)
        return data_new

    def _get_label(self, idx):
        return self.labels[os.path.basename(os.path.dirname(self.items_path[idx]))]

    def _get_triplet(self, idx):
        positive_list = []
        negative_list = []
        anchor_label = self._get_label(idx)

        for i in range(len(self.items_path)):
            if i == idx:
                continue
            i_label = self._get_label(i)
            if i_label == anchor_label:
                positive_list.append(self.items_path[i])
            else:
                negative_list.append(self.items_path[i])
        return positive_list, negative_list

    def __getitem__(self, idx):
        anchor = np.load(self.items_path[idx])
        anchor = self._get_data(anchor)
        
        if self.is_train:
            positive_list, negative_list = self._get_triplet(idx)
            positive = np.load(random.choice(positive_list))
            negative = np.load(random.choice(negative_list))
            positive = self._get_data(positive)
            negative = self._get_data(negative)
            return torch.from_numpy(anchor), torch.from_numpy(positive), torch.from_numpy(negative)
        else:
            anchor_label = self._get_label(idx)
            return torch.from_numpy(anchor), torch.from_numpy(anchor_label)

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone
