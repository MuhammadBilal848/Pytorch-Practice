import torch

# lets try and do gradient calculation
x = torch.tensor([1,2,3,4,5],dtype=float,requires_grad = True)
x = torch.randn(5,requires_grad = True)
print('printing x',x)

y = x + 2
print('printing y',y)

z = y*9*x
print('printing z' , z)
z = z.mean() # we have to compress values to a singular in order to calculate gradient.or else we have to
# pass equal length tensor in backword() function as param
print('printing z' ,z)
# now to calculate gradient(derivative) we can use backpropagation method in torch 
z.backward() # backprop of dz w.r.t dx: (dz/dx)
print('Gradients of dz/dx: ',x.grad) 

x = torch.tensor([1.,2.,3.],requires_grad=True)
print(x)
z = x**2
print(z)

# if z is not a singular value we will get an error, so in that case we have to pass another tensor in backward
# function as parameter
v = torch.tensor([0.1,0.2,0.3])
z.backward(v) # this will be (dz/dx) * v
print(x.grad)


###################### HOW TO STOP PYTORCH TO CREATE GRADIENT HISTORY ######################
# 1st way: var_name.require_grad_(False)
x1 = torch.randn(3,requires_grad = True)
x1.requires_grad_(False)
print(x1)

# 2nd way: var_name.detach()

x2 = torch.randn(3,requires_grad = True)
x2 = x2.detach()
print(x2)

# 3rd way: with torch.no_grad():


x3 = torch.randn(3,requires_grad = True)
with torch.no_grad():
    y = x3+2*4
    print(y)
