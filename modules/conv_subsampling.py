from torch import Tensor
import torch.nn as nn

'''
(roughly) reduces the input length (time dimension) by a factor of 4
input shape : (batch_size, input_channels, input_height, input_length)
output shape : (batch_size, input_channels*subsampling_factor, input_height//2//2, input_length//2//2)
'''



class ConvSubsampling(nn.Module):
    def __init__(self, input_channels: int, subsampling_factor: int) -> None:
        super(ConvSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*subsampling_factor, kernel_size=3, stride=2, bias=True)
        self.conv2 =nn.Conv2d(input_channels*subsampling_factor, input_channels*subsampling_factor, kernel_size=3, stride=2, bias=True)

        

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv1(inputs)
        x = nn.ReLU()(x) 
        x = self.conv2(x) 
        x  = nn.ReLU()(x) 
        output = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        return  output 