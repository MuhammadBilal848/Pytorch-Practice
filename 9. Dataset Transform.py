import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

# Data does not always come in its final processed form that is required for training machine learning algorithms. 
# We use transforms to perform some manipulation of the data and make it suitable for training.

class WineData(Dataset):
    def __init__(self, transforms = None):
        data = np.loadtxt('wine.csv',delimiter = ',',dtype = np.float32,skiprows = 1)
        self.x = data[:,1:]
        self.y = data[:,[0]]
        self.n_samples = data.shape[0]
        self.transform = transforms # we apply the transform here if it is not None

    def __getitem__(self, index):
        sample = self.x[index] , self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


# __call__ is a special method in python which is called when we call an object
    # exmaple of __call__ method is given below
    # class A:
    #     def __call__(self):
    #         print("call method is called")
    # a = A()
    # a() # this will call the __call__ method

# simple class to convert numpy array to tensors
class ConvertToTensor:
    def __call__(self,sample): 
        inputs , targets = sample
        return torch.from_numpy(inputs) , torch.from_numpy(targets)

class MulTransform: # this class will multiply the inputs by a factor
    def __init__(self,factor):
        self.factor = factor

    def __call__(self,sample):
        inputs , targets = sample
        inputs *= self.factor
        return inputs , targets


# example of using the ConvertToTensor class
# dataset = WineData(transforms = ConvertToTensor()) # we pass the transform object here
# features,labels = dataset[0]
# print(type(labels),type(features))

# example of using the MulTransform class and ConvertToTensor class
a = WineData()
f,l = a[0]
print('printing features before transformation',f)


composed = torchvision.transforms.Compose([ConvertToTensor(),MulTransform(5)])
d = WineData(transforms = composed)
features,labels  = d[0]
print(features)
print(type(labels),type(features))

