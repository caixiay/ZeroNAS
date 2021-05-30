import numpy as np
import torch
import torch.nn as nn
from operations_retrain import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NetworkRetrain(nn.Module):
    def __init__(self, opt, flag, num_layers, genotype):
        super(NetworkRetrain, self).__init__()
        self.num_layers = num_layers
        self.att_size = opt.attSize
        self.nz = opt.nz
        self.res_size = opt.resSize
        if flag == 'g':
            self.hidden_dim = [2 ** (7 + i) for i in range(num_layers)]
            self.initial_input_dims = [
                self.att_size,
                self.nz,
                self.att_size + self.nz
            ]
        else:
            self.hidden_dim = list(reversed([2 ** (7 + i) for i in range(num_layers - 1)]))
            self.hidden_dim.append(1)
            self.initial_input_dims = [
                self.att_size,
                self.res_size,
                self.att_size + self.res_size
            ]
        self.num_initial_input = len(self.initial_input_dims)
        self.input_node_idx_list = []

        # Generate all operations
        self.operation_list = nn.ModuleList()

        output_node_idx = self.num_initial_input
        num_operations = 0
        for operation_name, input_node_idx in genotype:

            # Save node idx for forward function
            self.input_node_idx_list.append(input_node_idx)

            # Input dim
            if input_node_idx < self.num_initial_input:
                in_dim = self.initial_input_dims[input_node_idx]
            else:
                in_dim = self.hidden_dim[input_node_idx-self.num_initial_input]

            # Output dim
            # output_flag = False
            # if output_node_idx == self.num_layers + self.num_initial_input - 1:  # output node
            #     out_dim = self.pre_traj_num * 2
            #     output_flag = True
            # else:
            #     out_dim = self.hidden_dim
            out_dim = self.hidden_dim[output_node_idx-self.num_initial_input]

            # Operation
            cur_operation = operation_dict_all[operation_name](in_dim, out_dim)
            # if output_flag:
            #     if in_dim == out_dim:
            #         cur_operation = operation_dict_same_dim_out[operation_name](in_dim, out_dim)
            #     else:
            #         cur_operation = operation_dict_diff_dim_out[operation_name](in_dim, out_dim)
            # else:
            #     if in_dim == out_dim:
            #         cur_operation = operation_dict_same_dim[operation_name](in_dim, out_dim)
            #     else:
            #         cur_operation = operation_dict_diff_dim[operation_name](in_dim, out_dim)

            self.operation_list.append(cur_operation)

            # Two operations per node
            num_operations = num_operations + 1
            if num_operations == 2:
                num_operations = 0
                output_node_idx = output_node_idx + 1
            # self.apply(weights_init)

    def forward(self, s_1, s_0):
        states = [s_0, s_1, torch.cat((s_0, s_1), dim=-1)]

        for i in range(self.num_layers):
            operation_1 = self.operation_list[i * 2]
            operation_2 = self.operation_list[i * 2 + 1]

            cur_state = operation_1(states[self.input_node_idx_list[i * 2]]) \
                        + operation_2(states[self.input_node_idx_list[i * 2 + 1]])

            states.append(cur_state)

        # Keep last layer output
        return states[-1]