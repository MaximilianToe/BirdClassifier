from torch  import Tensor
import torch.nn as nn   
from conformer_convolution_module import ConvolutionModule
from attention_block import MHSelfAttention
from feed_forward_module import FeedForwardModule

'''
Conformer Block that is part of the Conformer model
input shape : (batch_size, input_height, input_length)
output shape : (batch_size, input_height, input_length)
'''

class ConformerBlock(nn.Module):
    def __init__(self, input_height: int, n_head: int, kernel_size: int, dropout: float, device ) -> None:
        super(ConformerBlock, self).__init__()
        self.input_height = input_height
        self.ff1 = FeedForwardModule(input_height)
        self.ReLU = nn.ReLU()
        self.mhsa = MHSelfAttention(input_height = input_height, n_head=n_head, dropout=dropout, device=device)
        self.conv_block = ConvolutionModule(input_height, kernel_size) 
        self.ff2 =  FeedForwardModule(input_height) 
        self.ln = nn.LayerNorm(input_height)

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(1,2)
        x = 1/2*self.ff1(x) + x  
        x = x.transpose(1,2) 
        x = x + self.mhsa(x)
        x = x + self.conv_block(x)  
        x = x.transpose(1,2)
        x = 1/2*self.ff2(x) +x 
        output = self.ln(x).transpose(1,2)
        return  output 