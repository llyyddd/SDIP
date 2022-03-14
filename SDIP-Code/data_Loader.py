import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import copy

class MyCustomDataset(Dataset):
    def __init__(self,txt_path,label_format):
        '''
        :param txt_path: the file path of dataset
        :param label_format: Whether the class label starts from 1 or 0
        '''

        self.data=pd.read_table(txt_path, sep='  ', header=None, engine='python')
        #print(self.data)
        self.labels = np.asarray(self.data.iloc[:, 0])
        #print(self.labels)
        self.label_format=label_format


    def __getitem__(self, index):
        # Return the index-th sample of the training set (data-label)
        num = self.labels[index]
        # class label starts from 1
        if self.label_format==1:
           label = (self.labels[index] - 1).astype('int64')
        # class label starts from 0
        elif self.label_format==0:
           label = (self.labels[index]).astype('int64')


        data = np.asarray(self.data.iloc[index][1:])

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)
        #        if self.transforms is not None:
        #           data = self.transforms(data)
        return (data, label)


    def __len__(self):
        return len(self.data.index)

if __name__=='__main__':
    custom_dataset = MyCustomDataset('./data/TwoLeadECG_TRAIN.txt')
    # Define data loader
    dataset_loader = DataLoader(dataset=custom_dataset,
                                batch_size=10,
                                shuffle=False)
    print(len(dataset_loader))
    for data, labels in dataset_loader:
        print(data, labels)

