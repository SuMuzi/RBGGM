# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from tensorboard_logger import configure
from torch.autograd import Variable
from RDN import RDN
from RRDBNet import RRDB

n_feats = 64


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2  # inter_channel是in_channel的一半
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))  # 对g进行maxpool
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        # 这就是forward
        batch_size, C, H, W = x.shape
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = torch.softmax(f, dim=-1)  # softmax一下

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        # nonlocal module
        self.non_local = (
            NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()
        # self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


def swish(x):
    return x * torch.sigmoid(x)
    # return x * torch.tanh(F.softplus(x))  # mish


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])
        # 列表前面加星号作用是将列表解开成两个独立的参数，传入函数

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        # self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        # self.c_attention = SELayer(64, reduction=16)
        self.c_attention = CAM_Module()

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        Y = self.c_attention(y)
        return Y + x


class RG(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1, num_rcab=10):
        super(RG, self).__init__()
        self.module = [residualBlock() for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(n, n, k, stride=s, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class Upsampler(torch.nn.Module):
    def __init__(self, n_feats):
        super(Upsampler, self).__init__()
        # self.conv = nn.Conv2d(n_feats, 16 * n_feats, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(n_feats, 25 * n_feats, 2, stride=2, padding=0)
        self.shuffler = nn.PixelShuffle(5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.shuffler(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.selayer = CAM_Module()

        self.conv1 = nn.Conv2d(4, 64, 3, stride=1, padding=1)

        self.non_local1 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 2, sub_sample=False,
                                      bn_layer=False)

        self.non_local2 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 2, sub_sample=False,
                                      bn_layer=False)

        for i in range(self.n_residual_blocks):
            self.add_module('RRDB' + str(i + 1), RRDB(64))

        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), Upsampler(64))

        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 1, 1, padding=0)
        self.rdn = RDN()
        self.rrdb = RRDB(64)

    def forward(self, x):
        x = self.selayer(x)

        x = self.conv1(x)

        # x = self.non_local1(x)

        y = x.clone()

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('RRDB' + str(i + 1))(y)


        x = self.conv2(y) + self.gamma * x

        # x = self.non_local2(x)

        # x = res + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv4(self.conv3(x))

# D = Discriminator()
# print(D)
#
G = Generator(5, 2)
a = torch.randn(16,4,80,80)
print(G(a).shape)
# print(G)


