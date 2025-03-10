import torch
from torch import nn


class CA_Module(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(CA_Module, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.num_hidden = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.Conv_in = nn.Conv3d(in_channels=self.in_channel, out_channels=self.num_hidden,
                                 kernel_size=self.kernel_size, stride=1, padding=1)
        self.Conv_out = nn.Conv3d(in_channels=self.num_hidden, out_channels=self.in_channel,
                                  kernel_size=self.kernel_size, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.Conv_in(x)
        x = self.relu(x)
        x = self.Conv_out(x)
        x = self.sig(x)
        return x + 1

    @staticmethod
    def _check_num_hidden(num_hidden):
        if num_hidden != 1:
            raise ValueError("num_hidden != 1")


if __name__ == '__main__':
    input = torch.randn(3, 10, 32, 64, 64)  # b t c h w
    model = CA_Module(in_channel=10, kernel_size=3)
    output = model(input)
    print(output.size())
