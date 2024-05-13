from torch import Tensor
import torch.nn as nn

'''
Define the Multihead Self-Attention Module
input shape : (batch_size, input_height, input_length)
output shape : (batch_size, input_height, input_length)
'''

class MHSelfAttention(nn.Module):
    def __init__(self, input_height: int, n_head: int, dropout: float, device ) -> None:
        super(MHSelfAttention, self).__init__()
        self.d_height = input_height
        self.n_head = n_head
        if self.d_height % self.n_head != 0:
            raise ValueError(f"input_channels({self.d_height}) % n_head({self.n_head}) should be zero.")
        self.d_head = self.d_height // self.n_head
        self.porj_query = nn.Linear(self.d_height, self.d_height)
        self.proj_key = nn.Linear(self.d_height, self.d_height)
        self.proj_value = nn.Linear(self.d_height, self.d_height)
        self.mhsa = nn.MultiheadAttention(self.d_height, n_head, dropout, device)
        self.ln = nn.LayerNorm(self.d_height)



    def forward(self, input: Tensor) -> Tensor:
        x= input.transpose(1,2)
        x = self.ln(x)
        Q = self.porj_query(x)
        K = self.proj_key(x)
        V = self.proj_value(x)
        output, _ = self.mhsa(query=Q, key=K, value=V)
        output = nn.Dropout(0.1)(output)
        return  output.transpose(1,2) + input