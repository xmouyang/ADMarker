import math
import numpy as np
import torch
import os


class Multimodal_train_dataset():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id, num_of_samples):

        self.folder_path = "../AD-example-data/node{}/train_label_data/".format(node_id)
        y = np.load("../AD-example-data/node{}/train_label_data/label.npy".format(node_id))

        self.labels = y.tolist() #tolist
        self.labels = torch.tensor(self.labels).long()
        self.num_of_samples = num_of_samples


    def __len__(self):

        if self.num_of_samples < len(self.labels):
            return self.num_of_samples
        else:
            return len(self.labels)

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

        activity_label = self.labels[idx]

        return sensor_data1, sensor_data2, sensor_data3, activity_label


class Multimodal_test_dataset():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id):

        self.folder_path = "../AD-example-data/node{}/test_data/".format(node_id)
        y = np.load("../AD-example-data/node{}/test_data/label.npy".format(node_id))

        self.labels = y.tolist() #tolist
        self.labels = torch.tensor(self.labels).long()


    def __len__(self):
        return len(self.labels)

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

        activity_label = self.labels[idx]


        return sensor_data1, sensor_data2, sensor_data3, activity_label

def count_num_per_class(node_id, num_class, num_of_samples):

    original_label = np.load("../AD-example-data/node{}/train_label_data/label.npy".format(node_id))

    if num_of_samples < original_label.shape[0]:
        y = original_label[0:num_of_samples]
    else:
        y = original_label

    count_y = np.bincount(np.array(y).astype(int), minlength = num_class).astype(float)

    for idx in range(count_y.shape[0]):
        if count_y[idx] == 0:
            count_y[idx] = 0.5

    return count_y, y, len(y)



