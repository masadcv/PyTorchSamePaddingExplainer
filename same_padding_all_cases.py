import torch
import torch.nn as nn
import numpy as np

featuremap_size = 128

input = torch.rand(1, 1, featuremap_size, featuremap_size)

#######################################################################
print()
print("Even Kernel Tests:")

# equation from : https://stats.stackexchange.com/a/410270
# P = (((S-1)*F-S+K)/2), with K = kernel size, S = stride, F = input featuremap size
# even kernel size: P- = floor(P), P+ = ceil(P)
# odd kernel  size: P- = P+ = ceil(P)

for kernel_size in [x+1 for x in range(9)]:
    print("Kernel Size:", kernel_size)
    # kernel_size = 8
    stride = 1
    padding_minus = int(np.floor(((stride - 1) * featuremap_size - stride + kernel_size) / 2)) 
    padding_plus = int(np.ceil(((stride - 1) * featuremap_size - stride + kernel_size) / 2))

    if kernel_size % 2:
        padding_minus = padding_plus

    padlayer = nn.ZeroPad2d(padding=[padding_minus, padding_plus, 0, 0])
    layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, kernel_size], stride=1, bias=False)
    layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, kernel_size], stride=1, bias=False, padding="same", padding_mode="zeros")
    layer_w = np.random.rand(1, 1, 1, kernel_size).astype(np.float32)

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(layer_w))
        layer2.weight.copy_(torch.from_numpy(layer_w))
        
    output = layer(padlayer(input)).detach().cpu().numpy()
    output2 = layer2(input).detach().cpu().numpy()
    # print(input.shape)
    # print(output.shape)
    # print(output2.shape)

    print(np.sum(np.abs(output - output2)))
    print(np.allclose(output, output2))
    np.testing.assert_equal(output, output2)
    
    print()
    print()

    #######################################################################
    padlayer = nn.ZeroPad2d(padding=[0, 0, padding_minus, padding_plus])
    layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[kernel_size, 1], stride=1, bias=False)
    layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[kernel_size, 1], stride=1, bias=False, padding="same", padding_mode="zeros")
    layer_w = np.random.rand(1, 1, kernel_size, 1).astype(np.float32)

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(layer_w))
        layer2.weight.copy_(torch.from_numpy(layer_w))
        
    output = layer(padlayer(input)).detach().cpu().numpy()
    output2 = layer2(input).detach().cpu().numpy()
    # print(input.shape)
    # print(output.shape)
    # print(output2.shape)

    print(np.sum(np.abs(output - output2)))
    print(np.allclose(output, output2))
    np.testing.assert_equal(output, output2)

    print()
    print()
    
