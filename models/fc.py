import torch


class Fc(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Fc, self).__init__()
        
        self.r = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(in_features, 1024)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1024, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.r(x) - self.r(x-1)
        return x