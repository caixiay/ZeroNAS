import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FCReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLeakyReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCLeakyReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLeakyReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCBNLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNLeakyReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCBNReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCBNLeakyReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNLeakyReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCOut(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCOut, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.operation(x)



# All operation dict
operation_dict_all = {
    # 'fc': lambda in_dim, out_dim: FC(in_dim, out_dim),
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
    'fc_lrelu': lambda in_dim, out_dim: FCLeakyReLU(in_dim, out_dim),
    'fc_bn_relu': lambda in_dim, out_dim: FCBNReLU(in_dim, out_dim),
    'fc_bn_lrelu': lambda in_dim, out_dim: FCBNLeakyReLU(in_dim, out_dim),
    'fc_relu_d': lambda in_dim, out_dim: FCReLUdrop(in_dim, out_dim),
    'fc_lrelu_d': lambda in_dim, out_dim: FCLeakyReLUdrop(in_dim, out_dim),
    'fc_bn_relu_d': lambda in_dim, out_dim: FCBNReLUdrop(in_dim, out_dim),
    'fc_bn_lrelu_d': lambda in_dim, out_dim: FCBNLeakyReLUdrop(in_dim, out_dim),
}

operation_dict_diff_dim = {
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
    'fc_lrelu': lambda in_dim, out_dim: FCLeakyReLU(in_dim, out_dim),
}

operation_dict_same_dim = {
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
    'skip_connect': lambda in_dim, out_dim: Identity(),
}


operation_dict_same_dim_out = {
    'fc_out': lambda in_dim, out_dim: FCOut(in_dim, out_dim),
    'skip_connect': lambda in_dim, out_dim: Identity(),
}