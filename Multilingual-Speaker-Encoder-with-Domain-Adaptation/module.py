import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    """
        Graident reverse layer
    """
    @staticmethod
    def forward(ctx, x, lambd):
        lambd = torch.tensor(lambd, requires_grad=False)
        ctx.save_for_backward(lambd)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambd, = ctx.saved_tensors
        return (-lambd * grad_output, None)

def grad_reverse(x, lambd=1.0):
    return GradReverse().apply(x, lambd)

class MultiLayerNN(nn.Module):
    def __init__(self, indim, outdims, dropout=0.0):
        super(MultiLayerNN, self).__init__()
        self.layer = nn.Sequential()
        self.dropout = nn.Dropout(dropout, inplace=False)
        for i in range(len(outdims)):
            if i != len(outdims) - 1:
                self.layer.add_module('linear_'+str(i), LinearNorm(indim, outdims[i], w_init_gain='relu'))
                self.layer.add_module('dropout_'+str(i), self.dropout)
                self.layer.add_module('relu_'+str(i), nn.ReLU(inplace=True))
            else:
                self.layer.add_module('linear_'+str(i), LinearNorm(indim,  outdims[i], w_init_gain='sigmoid'))
            indim = outdims[i]
    def forward(self, x):
        x = self.layer(x)
        return x

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear', weight_norm=False):
        super(LinearNorm, self).__init__()
        if weight_norm:
            self.linear_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
        if self.linear_layer.bias is not None:
            torch.nn.init.zeros_(self.linear_layer.bias)

    def forward(self, x):
        x = self.linear_layer(x)
        return x

class ConvNorm2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, signal):
        return self.bn(self.conv(signal))

class ConvNorm1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', bn=True):
        super(ConvNorm1D, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
        self.bn = nn.BatchNorm1d(out_channels) if bn else None

    def forward(self, signal):
        if self.bn is not None:
            return self.bn(self.conv(signal))
        else:
            return self.conv(signal)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = ConvNorm2D(in_channels, out_channels, kernel_size=kernel_size, w_init_gain='relu')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvNorm2D(out_channels, out_channels, kernel_size=kernel_size, w_init_gain='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        out = self.relu(out)
        return out
