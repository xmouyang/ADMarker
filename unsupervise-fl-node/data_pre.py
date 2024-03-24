import math
import numpy as np
import torch
import os




class Multimodal_unlabel_dataset():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id, num_of_data):

        self.folder_path = "../AD-example-data/node{}/train_unlabel_data/".format(node_id)

        self.num_of_data = num_of_data


    def __len__(self):
        return self.num_of_data

    def __getitem__(self, idx):

        # print("idx:", idx)
        
        x1 = np.load(self.folder_path + "audio/" + "{}.npy".format(idx))
        x2 = np.load(self.folder_path + "depth/" + "{}.npy".format(idx))
        x3 = np.load(self.folder_path + "radar/" + "{}.npy".format(idx))

        self.data1 = x1.tolist() #concate and tolist
        self.data2 = x2.tolist() #concate and tolist
        self.data3 = x3.tolist()
        
        sensor_data1 = torch.tensor(self.data1) # to tensor
        sensor_data2 = torch.tensor(self.data2).float() # to tensor
        sensor_data3 = torch.tensor(self.data3) # to tensor

        sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        return sensor_data1, sensor_data2, sensor_data3



