import torch
import numpy as np

# to get an empty tensor(tensor is just like an array but it is supported by GPU Computation)
x = torch.empty(5,3)
print(x)
print(x.T) # we can also take transpose just like numpy


y = torch.ones(3,3)
print(y)
print(y.size())
print(y.shape)

# we can add two tensors just like numpy
a1 = torch.rand(3,3)
a2 = torch.rand(3,3)
print('Addition of two tensors:',a1+a2)
print('Addition of two tensors:',torch.add(a1,a2))

print('Multiplication of two tensors:',a1*a2)
print('Multiplication of two tensors:',torch.mul(a1,a2))

# we can also perfom inplace addition 
a2.add_(a1)
print(a2)

##################################################################################################
#         ANY OPERATION IN PYTORCH INVOLVING UNDERSCORE(_) WILL DO INPLACE OPERATION
##################################################################################################

# torch.add() for addition | For inplace a2.add_(a1)
# torch.sub() for subtraction | For inplace a2.sub_(a1) 
# torch.mul() for multiplication | For inplace a2.mul_(a1)
# torch.div() for division | For inplace a2.div_(a1)

# indexing can be done the same way as numpy
l = torch.rand(9,4)
print(l)
print(l[:,-1])
print(l[7:,2:])

print(l[6,3])
print(l[6,3].item()) # .item() is used to get the value directly.

# view() reshapes the tensor without copying memory, similar to numpy's reshape().
# lets say we wanna reshape (9,4) to (3,3,4)
print(l.view(3,3,4))


# we can convert tensor to numpy array
ttn = torch.rand(4,4)
ttn1 = ttn.numpy()
print(ttn,type(ttn))
print(ttn1,type)


# similarly we can convert numpy array to tensor

ttn = np.random.randint(0,9,(5,5))
ttn1 = torch.from_numpy(ttn)
print(ttn,type(ttn))
print(ttn1,type)


# we can set tensors to use GPU instance and do fast computations

if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5,device=device) # we have set the tensor to use cuda if it is available
    y = torch.empty(5)
    y = y.to(device) # another way to set tensor to cuda
    z = torch.mul(x,y)
    # thing to note here is we cant convert z to numpy using z.numpy() as numpy arrays support cpu operations
    # if we wanna convert the z(tensor) to numpy we first need to change device of z to cpu
    z = z.to('cpu')
else:
    print('GPU aint available!')


# another thing is that we can set the tensor parameters of gradient calculation to true if we are going to have
# future calculations of gradient(means if we wanna optimize any variable in the calculation, we need to set
# requires_grad to True)
x = torch.ones(10,requires_grad=True) # this 
print(x)
