import torch
import torch.nn as nn
from conformer_block import ConformerBlock
from conv_subsampling import ConvSubsampling

'''
Complete Conformer model
This combines the Convolutional Subsampling, Conformer Blocks and the final Linear layer to output a tensor of logits
input shape : (batch_size, input_channels, input_height, input_length)
output shape : (batch_size, num_classes)
'''



class Conformer(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 input_channels: int,
                 input_height: int, 
                 input_length: int,
                 num_heads: int,
                 num_conformer_blocks: int,
                 conv_kernel_size: int,
                 subsampling_factor: int,
                 device
                 ) -> None:
        super(Conformer, self).__init__()
        self.conv_subsampling = ConvSubsampling(input_channels=input_channels, subsampling_factor=subsampling_factor)
        self.conv_output_height = (((input_height-1)//2-1)//2)*subsampling_factor*input_channels
        self.conv_output_length = (((input_length-1)//2-1)//2)

        self.linear1 = nn.Linear(self.conv_output_height, self.conv_output_height) 
        self.dropout = nn.Dropout(0.1)
        self.conformer_blocks = nn.ModuleList([
             ConformerBlock(input_height=self.conv_output_height,
                            n_head=num_heads,
                            kernel_size=conv_kernel_size,
                            dropout=0.1,
                            device=device) for i in range(num_conformer_blocks)
         ])

        self.final_layer = nn.Linear(self.conv_output_length*self.conv_output_height, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv_subsampling(inputs)
        x = x.permute(0, 2, 1)
        x = (self.linear1(x)).permute(0, 2, 1) 
        x = self.dropout(x)
        for block in self.conformer_blocks:
            x = block(x)      
        m, _, __ = x.shape
        x = torch.reshape(x,(m, -1)) 
        y = self.final_layer(x)
        return y 

        