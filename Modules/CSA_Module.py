import torch
from torch import nn
from Modules.CA_Modules import CA_Module
from Modules.SA_Modules import SA_Module


def check_hidden_dim_num_layers(CSA_hidden_dim, CSA_num_layers):
    if len(CSA_hidden_dim) != CSA_num_layers:
        raise RuntimeError('len(CSA_hidden_dim) != CSA_num_layers !!!!')


def Cal_CSA_num(CA_bool, SA_bool):
    if CA_bool:
        CA_bool = 1
    else:
        CA_bool = 0
    if SA_bool:
        SA_bool = 1
    else:
        SA_bool = 0
    return SA_bool + CA_bool + 1


class CSA_Module(nn.Module):
    def __init__(self, in_channel, CSA_hidden_dim, CSA_num_layers,
                 kernel_size=3, channel_second=False, CA_bool=True, SA_bool=True):
        super(CSA_Module, self).__init__()
        self.in_channel = in_channel
        self.CSA_hidden_dim = CSA_hidden_dim
        self.CSA_num_layers = CSA_num_layers
        check_hidden_dim_num_layers(self.CSA_hidden_dim, self.CSA_num_layers)
        self.kernel_size = kernel_size
        self.channel_second = channel_second

        self.CA_bool = CA_bool
        self.SA_bool = SA_bool
        self.CSA_num = Cal_CSA_num(self.CA_bool, self.SA_bool)

        CSA_cell_list = []
        for i in range(0, self.CSA_num_layers):
            cur_input_dim = in_channel if i == 0 else self.CSA_hidden_dim[i - 1]
            CSA_cell_list.append(nn.Conv3d(in_channels=cur_input_dim, out_channels=self.CSA_hidden_dim[i],
                                           kernel_size=3, stride=1, padding=1))
            if self.CA_bool:
                CSA_cell_list.append(CA_Module(in_channel=self.CSA_hidden_dim[i], kernel_size=3))
            if self.SA_bool:
                CSA_cell_list.append(SA_Module(in_channel=self.CSA_hidden_dim[i], kernel_size=3))
        self.CSA_cell_list = nn.ModuleList(CSA_cell_list)
        self.Conv_last = nn.Conv3d(in_channels=self.CSA_hidden_dim[-1], out_channels=self.CSA_hidden_dim[-1],
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        if not self.channel_second:
            # b t c h w ---> b c t h w
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4).contiguous()
        for layer_idx in range(self.CSA_num_layers):
            conv_first = self.CSA_cell_list[layer_idx * self.CSA_num](input_tensor)
            if self.CA_bool:
                CA = self.CSA_cell_list[(layer_idx * self.CSA_num) + 1](conv_first)
            if self.SA_bool:
                SA = self.CSA_cell_list[(layer_idx * self.CSA_num) + self.CSA_num - 1](conv_first)
            if self.CSA_num == 3:
                input_tensor = conv_first * CA * SA
            if self.CSA_num == 2 and self.CA_bool:
                input_tensor = conv_first * CA
            if self.CSA_num == 2 and self.SA_bool:
                input_tensor = conv_first * SA
            if self.CSA_num == 1:
                input_tensor = conv_first
        output_tensor = self.Conv_last(input_tensor)
        # b c t h w ---> b t c h w
        output_tensor = output_tensor.permute(0, 2, 1, 3, 4).contiguous()
        return output_tensor


if __name__ == '__main__':
    input = torch.randn(3, 84, 1, 71, 73)  # b t c h w
    model = CSA_Module(in_channel=1, CSA_hidden_dim=[8], CSA_num_layers=1, CA_bool=True, SA_bool=True)
    print(model)
    output = model(input)
    print(output.size())
    # print(output)
