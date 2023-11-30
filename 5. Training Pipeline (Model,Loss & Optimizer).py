# 1. Model Design (input , output & forward pass)
# 2. Construct loss & optimzer
# 3. Training Loop
    # - forward pass
    # - backward pass
    # - update weights

import torch
import torch.nn as nn # neural network module of pytorch

# for pytorch we need to change the shape of the array from [1,2,3] to [[1],[2],[3]]
x = torch.tensor([[1],[2],[3],[4],[5]],dtype = torch.float32) # input
y = torch.tensor([[3],[5],[11],[7],[9]],dtype = torch.float32) # output

xtest = torch.tensor([2.0],dtype=torch.float32)

# here we define the initial weight
lr = 0.01
n_iters = 100

# define model
# we are doing a regression problem so we use Linear() method.
# it needs the input and the output size
# we know that we have to enter 1 number and in return we get output of 1 number also
model = nn.Linear(in_features=1 , out_features=1)

# loss
# MSE - (1(w*x - y)**2)/N
# we dont need to define custom loss , we can now use pytorch's loss
# here is the loss documentation : https://pytorch.org/docs/stable/nn.html#loss-functions
# some famous ones are: nn.L1Loss(MAE Loss) , nn.MSELoss , nn.CrossEntropyLoss
loss = nn.MSELoss()

# we also have to define optimizer
# here is the optimizer documentation : https://pytorch.org/docs/stable/optim.html
# we now have model's own parameter i.e. w and b , so we can use them
optimizer = torch.optim.SGD(model.parameters() , lr=lr) # weight initialization is now random

print(f'Prediction before training: f(5) = {model(xtest).item():.3f}')   

# training

for epoch in range(n_iters):

    # predictions
    ypred = model(x)

    # loss (both are usefull)
    l = loss(ypred,y)
    # l = loss(y,ypred)

    # gradients
    l.backward()

    # we dont need to update weights manually now
    optimizer.step()

    # we also have to empty the gradients after every iteration, as torch will write and accumulate gradients
    optimizer.zero_grad()

    if epoch % 1 == 0:
        w,b = model.parameters()
        print(f'epoch {epoch+1}: w = {w.item():.3f} : b = {b.item():.3f}, loss = {l:.8f}')



print(f'Expectation {2+3} vs Prediction after training: f(2) = {model(xtest).item():.3f}')   
print(f'Expectation {6+7} vs Prediction after training: f(6) = {model(torch.tensor([6.0],dtype=torch.float32)).item():.3f}')   
print(f'Expectation {9+10} vs Prediction after training: f(9) = {model(torch.tensor([9.0],dtype=torch.float32)).item():.3f}')   



############################################################################################################
#                                               Using Custom Class
############################################################################################################


x = torch.tensor([[1],[2],[3],[4],[5]],dtype = torch.float32) # input
y = torch.tensor([[3],[5],[11],[7],[9]],dtype = torch.float32) # output

xtest = torch.tensor([2.0],dtype=torch.float32)

lr = 0.01
n_iters = 100

# instead of using built in pytorch layer we can make our own class using builtin layer
class LinearRegression(nn.Module):
    def __init__(self,input_param , output_param):
        super(LinearRegression,self).__init__()
        # we can also use 'super().__init__()'
        self.linear = nn.Linear(input_param , output_param)

    def forward(self,x):
        return self.linear(x)
    
model = LinearRegression(1,1)

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters() , lr=lr) # weight initialization is now random

print(f'Prediction before training: f(5) = {model(xtest).item():.3f}')   

for epoch in range(n_iters):

    ypred = model(x)

    l = loss(ypred,y)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 1 == 0:
        w,b = model.parameters()
        print(f'epoch {epoch+1}: w = {w.item():.3f} : b = {b.item():.3f}, loss = {l:.8f}')



print(f'Expectation {2+3} vs Prediction after training: f(2) = {model(xtest).item():.3f}')   
print(f'Expectation {6+7} vs Prediction after training: f(6) = {model(torch.tensor([6.0],dtype=torch.float32)).item():.3f}')   
print(f'Expectation {9+10} vs Prediction after training: f(9) = {model(torch.tensor([9.0],dtype=torch.float32)).item():.3f}')   





