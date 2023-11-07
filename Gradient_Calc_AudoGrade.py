import torch

# lets try and do gradient calculation
# x = torch.tensor([1,2,3,4,5],dtype=float,requires_grad = True)
# x = torch.randn(5,requires_grad = True)
# print(x)

# y = x + 2
# print(y)

# z = y*9*x
# print(z)
# z = z.mean() # we have to compress values to a singular in order to calculate gradient.or else we have to
# # pass equal length tensor in backword() function as param
# print(z)
# # now to calculate gradient(derivative) we can use backpropagation method in torch 
# z.backward() # backprop of dz w.r.t dx: (dz/dx)
# print('Gradients of dz/dx: ',x.grad) 

x = torch.tensor([1.,2.,3.],requires_grad=True)
print(x)
z = x**2
print(z)
z = z.mean()
print(z)
z.backward()
print(x.grad)



