import torch
import torch.nn as nn
import numpy as np

input = torch.rand(1, 1, 10, 10)

#######################################################################
print()
print()
padlayer = nn.ZeroPad2d(padding=[0, 1, 0, 0])
layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 2], stride=1, bias=False)
layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 2], stride=1, bias=False, padding="same", padding_mode="zeros")
layer_w = np.random.rand(1, 1, 1, 2).astype(np.float32)
layer_w[..., 0] = -1
layer_w[..., 1] = 1

with torch.no_grad():
    layer.weight.copy_(torch.from_numpy(layer_w))
    layer2.weight.copy_(torch.from_numpy(layer_w))
    
output = layer(padlayer(input)).detach().cpu().numpy()
output2 = layer2(input).detach().cpu().numpy()
print(input.shape)
print(output.shape)
print(output2.shape)

print(np.sum(np.abs(output - output2)))
print(np.allclose(output, output2))


#######################################################################
print()
print()
padlayer = nn.ZeroPad2d(padding=[0, 0, 0, 1])
layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2, 1], stride=1, bias=False)
layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2, 1], stride=1, bias=False, padding="same", padding_mode="zeros")
layer_w = np.random.rand(1, 1, 2, 1).astype(np.float32)
layer_w[..., 0, :] = -1
layer_w[..., 1, :] = 1

with torch.no_grad():
    layer.weight.copy_(torch.from_numpy(layer_w))
    layer2.weight.copy_(torch.from_numpy(layer_w))
    
output = layer(padlayer(input)).detach().cpu().numpy()
output2 = layer2(input).detach().cpu().numpy()
print(input.shape)
print(output.shape)
print(output2.shape)

print(np.sum(np.abs(output - output2)))
print(np.allclose(output, output2))
