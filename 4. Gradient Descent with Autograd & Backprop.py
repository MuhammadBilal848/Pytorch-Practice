# import numpy as np

# ##########################################################################################################
# #                      here we will implement linear regression manually from scratch 
# ##########################################################################################################


# # eqn of linear reg , we are ignoring bias here
# # f = w * x

# x = np.array([1,2,3,4,5],dtype = np.float32) # input
# y = np.array([3,5,11,7,9],dtype = np.float32) # output


# # here we define the initial weight
# w = 0.0

# # define model
# def forward(x):
#     return w * x

# # loss
# # MSE - (1(w*x - y)**2)/N
# def loss(y,y_pred):
#     return ((y_pred-y)**2).mean()

# # gradient
# # dJ/dw = 1(2x (w*x -y))/ N
# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred-y).mean()


# print(f'Prediction before training: f(5) = {forward(5):.3f}')   

# # training
# lr = 0.01
# n_iters = 10

# for epoch in range(n_iters):

#     # predictions
#     ypred = forward(x)

#     # loss
#     l = loss(y,ypred)

#     # gradients
#     dw = gradient(x,y,ypred) # dw/dl

#     # updating weights
#     w -= lr * dw

#     if epoch % 1 == 0:
#         print(f'epoch {epoch+1}: w = {w:.3f} , loss = {l:.8f}')


# print(f'Prediction after training: f(5) = {forward(2):.3f}')   
# print(f'Prediction after training: f(5) = {forward(6):.3f}')   


##########################################################################################################
#                      here we will implement linear regression using pytorch 
##########################################################################################################


import torch

x = torch.tensor([1,2,3,4,5],dtype = torch.float32) # input
y = torch.tensor([3,5,11,7,9],dtype = torch.float32) # output


# here we define the initial weight
w = torch.tensor(0.0,requires_grad=True,dtype=torch.float32)

# define model
def forward(x):
    return w * x

# loss
# MSE - (1(w*x - y)**2)/N
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

# gradient
# dJ/dw = 1(2x (w*x -y))/ N
def gradient(x,y,y_pred):
    return torch.dot(2*x,y_pred-y).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')   

# training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):

    # predictions
    ypred = forward(x)

    # loss
    l = loss(y,ypred)

    # gradients
    l.backward()

    # updating weights , we donot want to keep track of gradients so we use no_grad
    with torch.no_grad():
        w -= lr * w.grad

    # we also have to empty the gradients after every iteration, as torch will write and accumulate gradients
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} , loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(2):.3f}')   
print(f'Prediction after training: f(5) = {forward(6):.3f}')   
