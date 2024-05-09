from torch import Tensor
import torch.nn as nn  

'''
Convoluation Module that is part of the ConformerBlock
input shape : (batch_size, input_channels, sequence_length)
output shape : (batch_size, input_channels, sequence_length)
'''
class ConvolutionModule(nn.Module):
    def __init__(self, input_channels: int, kernel_size: int ) -> None:
        super(ConvolutionModule, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape = input_channels)
        self.pwconv1 = nn.Conv1d(input_channels, 2*input_channels, padding='same', kernel_size = 1, bias=True)
        self.GLU = nn.GLU(dim =1)
        self.dwconv = nn.Conv1d(input_channels, input_channels, padding='same', kernel_size = kernel_size, bias=True)
        self.bn = nn.BatchNorm1d(input_channels)
        self.SiLU = nn.SiLU()
        self.pwconv2 = nn.Conv1d(input_channels, input_channels, padding='same', kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, input: Tensor) -> Tensor:
        #need to permute the input tensor to match the expected shape of the LayerNorm
        x = input.permute(0,2,1)
        x = self.ln(x).permute(0,2,1)
        #Reverse permutation after LayerNorm
        x = self.pwconv1(x)
        x = self.GLU(x)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.SiLU(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        return x + input 

