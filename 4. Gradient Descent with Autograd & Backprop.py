import numpy as np

# here we will implement linear regression from scratch

# eqn of linear reg , we are ignoring bias here
# f = w * x

x = np.array([1,2,3,4,5],dtype = np.float32) # input
y = np.array([3,5,7,9,11],dtype = np.float32) # output


# here we define the initial weight
w = 0.0

# define model
def forward(a):
    return w * x

# loss
# MSE - (1(w*x - y)**2)/N
def loss(y,y_pred):
    return ((y-y_pred)**2).mean()

# gradient
# dJ/dw = 1(2x (w*x -y))/ N
def gradient(x,y,y_pred):
    return np.dot(2*x,y-y_pred).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')   

# training
lr = 0.01
n_iters = 10

for epoch in range(n_iters):

    # predictions
    ypred = forward(x)

    # loss
    l = loss(y,ypred)

    # gradients
    dw = gradient(x,y,ypred)

    # updating weights
    w -= lr * dw

    if epoch % 1 == 0:





