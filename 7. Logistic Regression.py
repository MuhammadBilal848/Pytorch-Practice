# 1. Model Design (input , output & forward pass)
# 2. Construct loss & optimzer
# 3. Training Loop
    # - forward pass
    # - backward pass
    # - update weights

import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# preprocessing
bc = datasets.load_breast_cancer()
# print(dir(bc)) # to see all the data descriptors
x,y = bc.data , bc.target
print('shape of x: ', x.shape)
print('shape of y: ', y.shape)
nsamples , nfeatures = x.shape
print(nsamples , nfeatures)

# train test split
xtrain ,xtest ,ytrain ,ytest = train_test_split(x,y ,test_size=0.2 ,random_state=5)
print(xtrain.shape ,xtest.shape ,ytrain.shape ,ytest.shape )

# scale
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# numpy to tensor
xtrain = torch.from_numpy(xtrain.astype(np.float32))
xtest = torch.from_numpy(xtest.astype(np.float32))
ytrain = torch.from_numpy(ytrain.astype(np.float32))
ytest = torch.from_numpy(ytest.astype(np.float32))

# reshaping ytrain and ytest
ytrain = ytrain.view(ytrain.shape[0],1)
ytest = ytest.view(ytest.shape[0],1)

# model

class LogisticRegression(nn.Module):
    def __init__(self,inp_param):
        super().__init__()
        self.logreg = nn.Linear(inp_param,1)

    def forward(self,x):
        return  torch.sigmoid(self.logreg(x))
    

model = LogisticRegression(nfeatures)

# loss and optimization
loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = 0.01 )

epochs = 100

for i in range(epochs):
    
    # forward
    ypred = model(xtrain)

    # loss
    l = loss(ypred,ytrain)

    # backward
    l.backward()

    # update weights
    optimizer.step()

    # empty gradients
    optimizer.zero_grad()

    if True:
        w,b = model.parameters()
        print(f'epoch {i+1} : loss = {l:.8f}')


print('########################## TESTING ON TRAIN EXAMPLE #########################')
print('features: ' ,xtrain[5] , 'target: ', ytrain[5])
print(torch.round(model(xtrain[5])))


print('########################## TESTING ON TRAIN EXAMPLE #########################')
print('features: ' ,xtest[5] , 'target: ', ytest[5])
print(torch.round(model(xtest[5])))

with torch.no_grad():
    y_predicted = model(xtest)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])
    print(f'accuracy: {acc.item():.4f}')
