import torch
import torch.nn as nn
import torch.nn.functional as F
# from tensorboard_logger import configure
from torch.autograd import Variable

n_feats = 64


class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        ## gc module
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        ##  SE 模块
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            ## 左路分支
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)

            ## 右路分支
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)

            # 获取全局attention
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # [N, C, H, W]
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            # broadcast机制
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)

            out = out + channel_add_term
        return out


# Non-local module
class Nl(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None):
        super(Nl, self).__init__()
        # self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels

        self.f_key = (nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                                kernel_size=1, stride=1, padding=0))
        self.f_query = self.f_key
        self.f_value = (nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                  kernel_size=1, stride=1, padding=0))
        self.W = (nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                            kernel_size=1, stride=1, padding=0))
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        # sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        context += x
        return context


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=16, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        # nonlocal module
        # self.non_local = (
        #     NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))
        self.non_local = Nl(in_channels=in_feat, key_channels=inter_feat, value_channels=inter_feat)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        # batch_size, C, H, W = x.shape
        # H1 = int(H / 2)
        # W1 = int(W / 2)
        # nonlocal_feat = torch.zeros_like(x)

        # feat_sub_lu = x[:, :, :H1, :W1]
        # feat_sub_ld = x[:, :, H1:, :W1]
        # feat_sub_ru = x[:, :, :H1, W1:]
        # feat_sub_rd = x[:, :, H1:, W1:]
        #
        # nonlocal_lu = self.non_local(feat_sub_lu)
        # nonlocal_ld = self.non_local(feat_sub_ld)
        # nonlocal_ru = self.non_local(feat_sub_ru)
        # nonlocal_rd = self.non_local(feat_sub_rd)
        # nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        # nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        # nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        # nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd
        nonlocal_feat = self.non_local(x)

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




class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(n, in_channels, k, stride=s, padding=1)
        self.c_attention = CAM_Module()

    def forward(self, x):
        # y = self.relu(self.bn1(self.conv1(x)))
        # y = self.bn2(self.conv2(y))
        y = self.relu((self.conv1(x)))
        y = self.conv2(y)

        Y = self.c_attention(y)
        return Y + x


class RG(nn.Module):
    def __init__(self, in_channels=64, k=3, s=1, num_rcab=5):
        super(RG, self).__init__()
        self.module = [residualBlock(in_channels) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(in_channels, in_channels, k, stride=s, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x).mul(0.2)


class Upsampler(torch.nn.Module):
    def __init__(self, n_feats):
        super(Upsampler, self).__init__()
        # self.conv = nn.Conv2d(n_feats, 16 * n_feats, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(n_feats, 25 * n_feats, 4, stride=2, padding=1)
        self.shuffler = nn.PixelShuffle(5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.shuffler(self.conv(x)))




class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        # y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# D = Discriminator()
# print(D)
#
# G = Generator(5, 2)
# print(G)
