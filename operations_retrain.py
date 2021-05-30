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


class PhysicalForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PhysicalForward, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, 4),
            nn.BatchNorm1d(4),  # BN for fast training
        )

        self.out_dim = out_dim

    def predict_locations(self, parameters, in_frames_num=3, pre_frames_num=15, fps=30.0):
        # Generate t matrix
        t_vector = torch.arange(in_frames_num, pre_frames_num + in_frames_num, 1.0).cuda() / fps
        t_vector_square = torch.pow(t_vector, 2)
        t_matrix = torch.stack((torch.ones(pre_frames_num).cuda(),
                                t_vector,
                                t_vector_square))

        # Get x axis paremeter
        x_param = parameters[:, :2]
        x_param = torch.cat((x_param,
                             torch.ones(x_param.shape[0]).view(-1, 1).cuda() * 0), dim=1)
        x_locs_est = torch.mm(x_param, t_matrix)

        # Get y axis parameter
        y_param = parameters[:, 2:]

        y_param = torch.cat((y_param, torch.ones(y_param.shape[0]).view(-1, 1).cuda() * (-0.5) * 9.8), dim=1)
        y_locs_est = torch.mm(y_param, t_matrix)

        # Combine results
        return torch.cat((x_locs_est, y_locs_est), dim=1)

    def forward(self, x):
        parameters = self.operation(x)
        return self.predict_locations(parameters, pre_frames_num=self.out_dim // 2)


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