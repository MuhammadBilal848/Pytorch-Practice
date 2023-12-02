import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineData(Dataset):
    def __init__(self):
        data = np.loadtxt('wine.csv',delimiter = ',',dtype = np.float32,skiprows = 1)
        self.x = data[:,1:]
        self.y = data[:,[0]]
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index] , self.y[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = WineData()
print(dataset[5])



# we can also use Dataloader

dataloader = DataLoader(dataset=dataset ,batch_size = 4 , shuffle = True) # this is a dataloader object
data_iter = iter(dataloader)
print(next(data_iter))
