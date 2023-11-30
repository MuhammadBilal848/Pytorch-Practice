# 1. Model Design (input , output & forward pass)
# 2. Construct loss & optimzer
# 3. Training Loop
    # - forward pass
    # - backward pass
    # - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# data creation
x_n , y_n = datasets.make_regression(n_samples=100,n_features=2,random_state=4)

x , y = datasets.make_regression(n_samples=100,n_features=2,random_state=4)
y = y.reshape(100,1)
print(x.shape,y.shape)
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32)) # right now y is 

# model

n_samples = 100
n_features = 2 # we have 2 features 
input_size = n_features
output = 1 # we get one number as output

model = nn.Linear(in_features=input_size , out_features=output)


# loss & optimizer

lr = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = lr)


# training loop

for epoch in range(200):

    # forward pass

    ypred = model(x)
    l = loss(ypred,y)

    # backward pass
    l.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()


    if True:
        w,b = model.parameters()
        print(f'epoch {epoch+1}: w1 = {w[0][0].item():.3f} : w2 = {w[0][1].item():.3f} : b = {b.item():.3f}, loss = {l:.8f}')

print(x[5],y[5])

xtest = torch.tensor([ 1.69235772,-1.11281215],dtype = torch.float32)
res = model(x[5]).detach().item()
print(res)

predicted = model(x).detach().numpy()

plt.plot(x_n, y_n, marker='o', linestyle='')  # 'o' for circles, no line connecting them
plt.plot(x_n, predicted, marker='x', linestyle='')  # 'x' for crosses, no line connecting them
plt.show()
