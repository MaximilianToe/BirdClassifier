import torch
import torch.nn as nn

'''
input shape : (batch_size, input_length)
output shape : (batch_size, input_length)
'''

class FeedForwardModule(nn.Module):
    def __init__(self, input_length: int) -> None:
        super(FeedForwardModule, self).__init__()
        self.ln = nn.LayerNorm(input_length)
        self.linear1 = nn.Linear(input_length, input_length*4) 
        self.SiLU = nn.SiLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(input_length*4, input_length) 
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln(input)
        x = self.linear1(x)
        x = self.SiLU(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x) 
        return x 