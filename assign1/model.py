import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(3,512)
        self.fc1 = nn.utils.weight_norm(self.fc1)
        self.fc2 = nn.Linear(512,512)
        self.fc2 = nn.utils.weight_norm(self.fc2)
        self.fc3 = nn.Linear(512,512)
        self.fc3 = nn.utils.weight_norm(self.fc3)
        self.fc4 = nn.Linear(512,509)
        self.fc4 = nn.utils.weight_norm(self.fc4)
        self.fc5 = nn.Linear(512,512)
        self.fc5 = nn.utils.weight_norm(self.fc5)
        self.fc6 = nn.Linear(512,512)
        self.fc6 = nn.utils.weight_norm(self.fc6)
        self.fc7 = nn.Linear(512,512)
        self.fc7 = nn.utils.weight_norm(self.fc7)
        self.fc8 = nn.Linear(512,1)
        self.fc8 = nn.utils.weight_norm(self.fc8)
        self.th  = nn.Linear(1,1)

        self.dropout = nn.Dropout(dropout_prob)
        self.use_tanh = use_tanh
    
    # input: N x 3
    def forward(self, input):

        z1 = self.fc1(input)
        a1 = F.leaky_relu(z1)
        a1 = self.dropout(a1)
        z2 = self.fc2(a1)
        a2 = F.leaky_relu(z2)
        a2 = self.dropout(a2)
        z3 = self.fc3(a2)
        a3 = F.leaky_relu(z3)
        a3 = self.dropout(a3)
        z4 = self.fc4(a3)
        z4 = torch.cat([z4, input], dim=1)
        a4 = F.leaky_relu(z4)
        a4 = self.dropout(a4)
        z5 = self.fc5(a4)
        a5 = F.leaky_relu(z5)
        a5 = self.dropout(a5)
        z6 = self.fc6(a5)
        a6 = F.leaky_relu(z6)
        a6 = self.dropout(a6)
        z7 = self.fc7(a6)
        a7 = F.leaky_relu(z7)
        a7 = self.dropout(a7)
        z8 = self.fc8(a7)
        a8 = F.leaky_relu(z8)
        a8 = self.dropout(a8)

        if self.use_tanh:
            z9 = self.th(a8)
            y = F.tanh(z9)
        else:
            y = a8

        return y
