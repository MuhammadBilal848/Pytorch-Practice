import torch


##################################### WATCH THIS VIDEO #######################################
# https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4
##################################### WATCH THIS VIDEO #######################################


x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0 , requires_grad=True) # since we need to calculate gradient of w so we set requires_grad = True,
# w here is the weight, as we know in n.n weights change in every backward pass



y_hat = w*x
loss = (y_hat-y)**2 # squared loss
print(loss)

# now to calculate gradient
loss.backward()
print(w.grad)


# now we can update weights
# next forward and backward pass and so onnnnnnnnnn.
