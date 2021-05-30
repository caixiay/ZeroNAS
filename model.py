import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_AC_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_AC_2HL_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_2HL_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, x):
        h = self.dropout(self.lrelu(self.fc1(x)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_3HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_3HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.fc4(h)
        return h

class MLP_2HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h

class MLP_2HL_Dropout_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_Dropout_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h

class MLP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class MLP_2HL_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.relu(self.fc3(h))
        return h

class MLP_3HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return h

class MLP_2HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.relu(self.fc2(h))
        return h

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        # self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        # h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        # self.fc3 = nn.Linear(opt.ndh//2, 1)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_2048_1024_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2048_1024_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc3 = nn.Linear(1024, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        #self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h


class MLP_SKIP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_SKIP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc_skip = nn.Linear(opt.attSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        #h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc2(h))
        h2 = self.fc_skip(att)
        return h+h2



class MLP_SKIP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_SKIP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc_skip = nn.Linear(opt.attSize, opt.ndh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h2 = self.lrelu(self.fc_skip(att))
        h = self.fc2(h+h2)
        return h

class MLP_G_up_down(nn.Module):
    def __init__(self, opt):
        super(MLP_G_up_down, self).__init__()
        # self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        # self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        # self.lrelu = nn.LeakyReLU(0.2, True)
        # #self.prelu = nn.PReLU()
        # self.relu = nn.ReLU(True)
        #
        # self.apply(weights_init)

        self.ch = 64
        self.bottom_width = 4
        self.l1 = nn.Linear(opt.attSize + opt.nz, (self.bottom_width ** 2) * self.ch)
        self.cell1 = Cell(self.ch, self.ch, 'nearest', num_skip_in=0, short_cut=True)
        self.cell2 = Cell(self.ch, self.ch, 'bilinear', num_skip_in=1, short_cut=True)
        self.cell3 = Cell(self.ch, self.ch, 'nearest', num_skip_in=2, short_cut=False)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )
        # self.fc = nn.Linear(3072, opt.resSize)
        # self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(3072, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        # h = self.lrelu(self.fc1(h))
        # h = self.relu(self.fc2(h))
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)
        output1 = output.view(output.size(0), -1)
        output2 = self.lrelu(self.fc1(output1))
        output3 = self.relu(self.fc2(output2))

        return output3

UP_MODES = ['nearest', 'bilinear']
NORMS = ['in', 'bn']

class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, ksize=3, num_skip_in=0, short_cut=False, norm=None):
        super(Cell, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize//2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        h = nn.ReLU()(residual)
        h = F.upsample(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.upsample(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.upsample(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out


class MLP_G_S1(nn.Module):
    def __init__(self, opt):
        super(MLP_G_S1, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.nz+opt.attSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.ngh)
        self.fc_out = nn.Linear(opt.ngh, opt.resSize)
        self.bn = nn.BatchNorm1d(opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h1 = self.lrelu(self.fc1(att))
        h2 = self.lrelu(self.fc2(h))
        # h3 = self.bn(self.lrelu(h1))
        # h4 = self.lrelu(h2)
        # h_out = self.fc_out(torch.cat((self.fc3(h1), self.fc4(h2)), dim=1))
        h_out = self.relu(self.fc_out(h2))
        return h_out

class MLP_CRITIC_S1(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC_S1, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.resSize+opt.attSize, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, opt.ndh)
        self.fc_out = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(opt.ndh)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h1 = self.lrelu(self.fc1(att))
        h2 = self.lrelu(self.fc2(h))
        h = self.fc_out(self.bn(self.lrelu(self.fc3(h1)))+self.relu(self.fc4(h2)))
        return h

class MLP_G_S2(nn.Module):
    def __init__(self, opt):
        super(MLP_G_S2, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.nz+opt.attSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.ngh)
        self.fc5 = nn.Linear(opt.ngh, opt.ngh)
        self.fc6 = nn.Linear(opt.ngh, opt.ngh)
        self.fc_out = nn.Linear(opt.ngh, opt.resSize)
        self.bn = nn.BatchNorm1d(opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h1 = self.lrelu(self.fc1(att))
        h2 = self.lrelu(self.fc2(h))
        h3 = self.fc3(h2)+self.lrelu(self.fc4(h1))
        h4 = self.fc5(h1)+self.fc6(h2)
        # h_out = self.fc_out(torch.cat((self.fc3(h1), self.fc4(h2)), dim=1))
        h_out = self.relu(self.fc_out(h3+h4))
        return h_out

class MLP_CRITIC_S2(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC_S2, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.resSize+opt.attSize, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, opt.ndh)
        self.fc5 = nn.Linear(opt.ndh, opt.ndh)
        self.fc6 = nn.Linear(opt.ndh, opt.ndh)
        self.fc_out = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(opt.ndh)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h1 = self.lrelu(self.fc1(att))
        h2 = self.lrelu(self.fc2(h))
        h3 = self.fc3(h1)+self.bn(self.sigmoid(self.fc4(h2)))
        h4 = self.bn(self.lrelu(self.fc5(h1)))+h2
        h_out = self.fc_out(h3+h4)
        return h_out

class MLP_G_CONV(nn.Module):
    def __init__(self, opt):
        super(MLP_G_CONV, self).__init__()
        self.bottom_width = 4
        self.ch = 64
        self.kernel = 3
        self.fc1 = nn.Linear(opt.attSize + opt.nz, (self.bottom_width ** 2) * self.ch)
        # self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.c1 = nn.Conv2d(self.ch, self.ch, self.kernel, padding=self.kernel//2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h)).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = F.upsample(h, scale_factor=2, mode='nearest')
        h = self.c1(h)
        h = self.relu(self.fc2(h.view(h.size(0), -1)))
        return h

class MLP_G_CONV2(nn.Module):
    def __init__(self, opt):
        super(MLP_G_CONV2, self).__init__()
        self.bottom_width = 4
        self.ch = 64
        self.kernel = 3
        self.fc1 = nn.Linear(opt.attSize + opt.nz, (self.bottom_width ** 2) * self.ch)
        # self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.c1 = nn.Conv2d(self.ch, self.ch, self.kernel, padding=self.kernel//2)
        self.c2 = nn.Conv2d(self.ch, self.ch, 5, padding=2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h)).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = F.upsample(h, scale_factor=2, mode='nearest')
        h = self.c1(h)
        h = self.c2(h)
        h = self.relu(self.fc2(h.view(h.size(0), -1)))
        return h

class MLP_CRITIC_CONV(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC_CONV, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

# A layer of all operations
class MixedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, operation_dict):
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
        for operation in operation_dict.keys():
            # Create corresponding layer
            layer = operation_dict[operation](input_dim, output_dim)
            self.layers.append(layer)

    def forward(self, x, weights):
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        res = sum(res)

        return res

from operations import *

# A network with mixed layers
class Network(nn.Module):
    def __init__(self, num_layers, initial_input_dims, hidden_dim):
        super(Network, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.operation_name_list = []
        self.initial_input_dims = initial_input_dims
        self.num_initial_input = len(self.initial_input_dims)

        # Generate all the mixed layer
        for i in range(self.num_layers):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    layer = MixedLayer(self.initial_input_dims[j], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))

                else:  # Middle layers
                    layer = MixedLayer(self.hidden_dim[j-self.num_initial_input], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))


    def forward(self, s_0, s_1, edge_weights, operation_weights):
        states = [s_0, s_1, torch.cat((s_0, s_1), dim=-1)]
        offset = 0

        # Input from all previous layers
        for i in range(self.num_layers):
            s = sum(
                edge_weights[i][j] * self.layers[offset + j](cur_state, operation_weights[offset + j]) for j, cur_state
                in enumerate(states))
            offset += len(states)
            states.append(s)

        # Keep last layer output
        return states[-1]

    def get_operation_name_list(self):
        return self.operation_name_list

class MLP_search(nn.Module):
    def __init__(self, opt, flag, num_layers = 5):
        super(MLP_search, self).__init__()
        # self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        # self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        # self.lrelu = nn.LeakyReLU(0.2, True)
        # self.relu = nn.ReLU(True)
        # self.apply(weights_init)
        self.num_layers = num_layers
        self.att_size = opt.attSize
        self.nz = opt.nz
        self.res_size = opt.resSize
        if flag=='g':
            self.hidden_dim = [2**(7+i) for i in range(num_layers)]
            self.initial_input_dims = [
                self.att_size,
                self.nz,
                self.att_size + self.nz
            ]
        else:
            self.hidden_dim = list(reversed([2 ** (7 + i) for i in range(num_layers-1)]))
            self.hidden_dim.append(1)
            self.initial_input_dims = [
                self.att_size,
                self.res_size,
                self.att_size + self.res_size
            ]
        print('self.hidden_dim', self.hidden_dim)


        self.num_initial_input = len(self.initial_input_dims)
        self.network = Network(self.num_layers, self.initial_input_dims, self.hidden_dim)
        # Get operation list
        self.operation_name_list = self.network.get_operation_name_list()

        # Alpha list for each operation
        self.operation_alpha_list = []
        for i in range(len(self.operation_name_list)):
            # print('operation', i, len(self.operation_name_list[i]))
            self.operation_alpha_list.append(
                Variable(1e-3*torch.randn(len(self.operation_name_list[i])).cuda(), requires_grad=True)
            )

        # Alpha list for each edge
        self.edge_alpha_list = []
        for i in range(self.num_layers):
            # print('edge', i, i + self.num_initial_input)
            self.edge_alpha_list.append(
                Variable(1e-3*torch.randn(i + self.num_initial_input).cuda(), requires_grad=True)
            )

        # self.apply(weights_init)
        # # Initialize alphas to smaller value
        # with torch.no_grad():
        #     for alpha in self.edge_alpha_list:
        #         alpha.mul_(1e-1)
        #
        # with torch.no_grad():
        #     for alpha in self.operation_alpha_list:
        #         alpha.mul_(1e-1)

    def forward(self, noise, att):
        # h = torch.cat((noise, att), 1)
        # h = self.lrelu(self.fc1(h))
        # # h = self.lrelu(self.fc2(h))
        # h = self.relu(self.fc2(h))
        h = self.network(att, noise, self.edge_weights(), self.operation_weights())
        return h
    def arch_parameters(self):
        return self.operation_alpha_list + self.edge_alpha_list

    def edge_weights(self):
        return [F.softmax(alpha, dim=-1) for alpha in self.edge_alpha_list]

    def operation_weights(self):
        return [F.softmax(alpha, dim=-1) for alpha in self.operation_alpha_list]

    # def edge_weights_masked(self):
    #     if self.training:
    #         return [RandomMask.apply(weight, 2) for weight in self.edge_weights()]
    #     else:
    #         mask_list = []
    #         for weight in self.edge_weights():
    #             # max_idx = torch.argsort(weight, descending=True)[:2]
    #             _, max_idx = weight.sort(0, descending=True)[:2]
    #             mask = torch.zeros_like(weight)
    #             mask[max_idx] = 1.0
    #             mask_list.append(mask)
    #         return mask_list
    #
    # def operation_weights_masked(self):
    #     if self.training:
    #         return [RandomMask.apply(weight, 1) for weight in self.operation_weights()]
    #     else:
    #         mask_list = []
    #         for weight_idx, weight in enumerate(self.operation_weights()):
    #             # sorted_idxs = torch.argsort(weight, descending=True)
    #             _, sorted_idxs = weight.sort(0, descending=True)
    #             max_idx = sorted_idxs[0]
    #             mask = torch.zeros_like(weight)
    #             mask[max_idx] = 1.0
    #             mask_list.append(mask)
    #
    #         return mask_list

    def get_cur_genotype(self):
        edge_weights = [weight.data.cpu().numpy() for weight in self.edge_weights()]
        operation_weights = [weight.data.cpu().numpy() for weight in self.operation_weights()]
        gene = []
        n = self.num_initial_input
        start = 0
        for i in range(self.num_layers):  # for each node
            end = start + n
            W = operation_weights[start:end].copy()

            best_edge_idx = np.argsort(edge_weights[i])[::-1]  # descending order

            for j in best_edge_idx[:2]:  # pick two best
                k_best = None
                for k in range(len(W[j])):  # get best ops for j->i
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((self.operation_name_list[start + j][k_best], j))  # save for plot
            start = end
            n += 1
        return gene

# class RandomMask(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, weight, num_masked_weights):
#         # ctx.save_for_backward(weight)
#         ctx.num_masked_weights = num_masked_weights
#         # Sample a weight
#         picked_idx = torch.multinomial(weight, num_masked_weights)
#         # masked_weight = torch.zeros(len(weight), requires_grad=True).to(device)
#         masked_weight = torch.zeros(len(weight)).cuda()
#         masked_weight[picked_idx] = 1.0
#
#         return masked_weight
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # weight, = ctx.saved_tensors
#         return grad_output.clone(), None