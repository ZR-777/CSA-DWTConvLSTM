import torch
from torch import nn


class SA_Module(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(SA_Module, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.num_hidden = in_channel
        self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.Conv1 = nn.Conv3d(in_channels=self.in_channel, out_channels=self.num_hidden,
                               kernel_size=self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.Conv2 = nn.Conv3d(in_channels=self.num_hidden, out_channels=self.in_channel,
                               kernel_size=self.kernel_size, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        temp = self.avg_pool(x)
        temp = self.Conv1(temp)
        temp = self.relu(temp)
        temp = self.Conv2(temp)
        temp = self.sig(temp).expand_as(x)
        return temp + 1


if __name__ == '__main__':
    input = torch.randn(3, 32, 10, 71, 73)  # b c t h w输入
    model = SA_Module(32, 3)
    output = model(input)
    print(output.size())
