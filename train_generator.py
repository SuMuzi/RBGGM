# import torch
# import torch.nn as nn
# import numpy as np
# from load_data_tg import TrainDataset
# from model import FSRCNN
# from generator_nl import Generator
# from DBPN import Net
# from RDN import RDN
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.optim as optim
# import os
# import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# import argparse
# from tqdm import tqdm
# from utils import AverageMeter, calc_psnr
# from csi_far_pod import *
# import torch.nn.functional as F
# import scipy.ndimage
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cudnn.benchmark = True
#
# batch_size = 64
# epochs = 100
# learning_rate = 0.0001
# checkpoint_path = 'G:/new_try/SRCNN/checkpoints_tp'
# if not os.path.exists(checkpoint_path):
#     os.mkdir(checkpoint_path)
#
# # a = [0.0000, 228.0563, 237.6905, 244.1756, 252.3060, -3.4042, -4.3378,
# #      -1.9270, 4.0818, 0.0000]
# # b = [814.4700, 278.2337, 297.9331, 308.9454, 319.3127, 106.6483,
# #      112.2740, 110.3443, 110.3622, 6761.0000]
# # a = [0.0000, 225.3556, 234.5274, 241.9779, -12.0390, -7.6769, -5.4955,
# #      0.0000]
# # b = [438.6700, 277.9438, 294.7638, 306.2963, 126.9618, 145.7404,
# #      146.5819, 6761.0000]
# # a = [ -7.5735, 251.7767, 253.3911, 252.4786, 234.4912, 234.8650, 236.0145,
# #         241.9779, 243.0362, 243.7981, 225.3570, 226.1343, 226.5610,  -0.5451,
# #          -0.9199,  -1.5364,  -4.6544,  -6.9894,  -4.0990,  -2.5770,  -5.6185,
# #          -4.0723,  -7.9286,  -8.4364,  -7.7853, -55.0000]
# # b = [ 363.6550,  315.9071,  321.6161,  323.1270,  294.7142,  301.0461,
# #          301.5699,  306.1771,  311.5555,  313.3928,  277.9315,  278.0332,
# #          279.6116,  142.9232,  111.1677,  126.8723,  139.7725,  116.4831,
# #          122.8586,  142.4455,  116.6583,  122.4874,  126.9901,  122.3413,
# #          124.0924, 5293.0000]
# a = [ -7.5735, 251.7767, 253.3911, 252.4786, 234.4912,  -0.5451,  -0.9199,
#          -1.5364,  -4.6544, -55.0000]
# b = [ 363.6550,  315.9071,  321.6161,  323.1270,  294.7142,  142.9232,
#          111.1677,  126.8723,  139.7725, 5293.0000]
# c = [b[i] - a[i] for i in range(len(a))]
# transform1 = transforms.Compose([
#     transforms.Normalize(mean=a,
#                          std=c)
#     # transforms.Normalize(mean=[0.5], std=[0.5])
# ])
# transform2 = transforms.Compose([
#     transforms.Normalize(mean=[0], std=[438.6700])
#     # transforms.Normalize(mean=[0.5], std=[0.5])
# ])
#
# max_g = 438.6700
#
# train_data = TrainDataset('G:/downscale/gpm', 'G:/new_try/gpm_data', 'G:/new_try/tem_0', 'G:/new_try/relative_hu_0',
#                           'G:/new_try/geopotential', transform1=transform1, transform2=transform2)
# train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# print(len(train_data))
#
# test_data = TrainDataset('G:/downscale/gpm_test', 'G:/new_try/gpm_test', 'G:/new_try/tem_test_0',
#                          'G:/new_try/r_h_test_0',
#                          'G:/new_try/geopotential_test', transform1=transform1, transform2=transform2)
# test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# img_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
# print(len(test_data))
#
# torch.manual_seed(1)
#
#
# # 一种loss
# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#     def forward(self, logits, targets, t):
#         num = targets.size(0)
#         smooth = 0.0001
#
#         # probs = F.sigmoid(logits)
#         m1 = logits.view(num, -1)
#         m1 = torch.where(m1 >= t, 1, 0)
#         m2 = targets.view(num, -1)
#         m2 = torch.where(m2 >= t, 1, 0)
#         intersection = (m1 * m2)
#
#         score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - 2 * score.sum() / num
#         return score
#
#
# criterion = nn.MSELoss()
# # criterion = nn.L1Loss()
#
# # model = Net(4, 16, 64)
# model = Generator(10, 2)
# # model = RDN().to(device)
#
# # if torch.cuda.device_count() > 1:
# #     print("Use", torch.cuda.device_count(), 'gpus')
# #     model = nn.DataParallel(model)
#
# model = model.to(device)
#
# print(model)
# epoch_loss = []
#
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#
# for epoch in range(epochs):
#     model.train()
#     train_loss = []
#     epoch_losses = AverageMeter()
#
#     with tqdm(total=(len(train_data) - len(train_data) % epochs), position=0) as t:
#         t.set_description('epoch:{}/{}'.format(epoch, epochs - 1))
#
#         for idx, (lr, hr) in enumerate(train_data_loader):
#             # print(lr.shape, hr.shape)
#             lr = lr.to(device)
#             hr = hr.to(device)
#
#             # lr = F.interpolate(hr, size=80, mode='bicubic',align_corners=True)
#
#             sr = model(lr)
#
#             hr_2 = hr.cpu().numpy()
#             hr_2 = np.array([scipy.ndimage.zoom(hr_2[i,0,:,:], 0.5) for i in range(len(hr_2))])
#             hr_2 = np.expand_dims(hr_2, 1)
#             hr_2 = torch.from_numpy(hr_2).to(device)
#
#             loss1 = criterion(sr[1], hr)  # + 0.001 * SoftDiceLoss()(sr * max_g, hr * max_g, 0.1)
#             loss2 = criterion(sr[0], hr_2)
#             loss = loss1+loss2
#
#             train_loss.append(loss.item())
#
#             epoch_losses.update(loss.item(), len(lr))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
#             t.update(len(lr))
#
#     train_loss = np.average(train_loss)
#     epoch_loss.append(train_loss)
#     scheduler.step()
#
#     print('train_loss: {}'.format(train_loss))
#     torch.save(model, os.path.join(checkpoint_path, 'epoch_{}.pth'.format(epoch)))
#     np.save(os.path.join(checkpoint_path, 'loss'), epoch_loss)
#     ## 测试效果
#     model.eval()
#     epoch_psnr = AverageMeter()
#     pod = AverageMeter()
#     csi = AverageMeter()
#     far = AverageMeter()
#     total_loss = []
#
#     for data in test_data_loader:
#         lr, hr = data
#
#         lr = lr.to(device)
#         hr = hr.to(device)
#
#         with torch.no_grad():
#             pre = model(lr)
#             pre = pre[1]
#             loss = criterion(pre.detach().cpu(), hr.detach().cpu())
#             total_loss.append(loss)
#
#             epoch_psnr.update(calc_psnr(pre, hr), len(lr))
#
#             pre = pre[0, 0, :, :].detach().cpu().numpy()
#             obs = hr[0, 0, :, :].detach().cpu().numpy()
#
#             pre = pre * max_g
#             obs = obs * max_g
#
#             csi.update(CSI(obs=obs, pre=pre, threshold=0.1), len(lr))
#             pod.update(POD(obs, pre, threshold=0.1), len(lr))
#             far.update(FAR(obs, pre, threshold=0.1), len(lr))
#
#     total_loss = np.average(total_loss)
#
#     print('eval psnr: {:.2f}  val_loss: {}'.format(epoch_psnr.avg, total_loss))
#     print('csi:{}  pod:{}  far:{}'.format(csi.avg, pod.avg, far.avg))
#     print('\n')
import torch
import torch.nn as nn
from load_data_tg import TrainDataset
from generator_trmm import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from tqdm import tqdm
from utils import AverageMeter, calc_psnr
from csi_far_pod import *
import torch.nn.functional as F
from Srsp import Get_gradient_nopadding
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 16
epochs = 200
learning_rate = 0.0001  # 小一点?
localtime = time.localtime(time.time())
tm = str(localtime[0])+str(localtime[1])+str(localtime[2])+str(localtime[3])+str(localtime[4])+str(localtime[5])
checkpoint_path = os.path.join('G:/SQG/after3131/SRCNN/checkpoints_clnla',tm)
#checkpoint_path = 'G:/new_try/SRCNN/checkpoints_clnla'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# a = [  0.0000, 244.2279,  -0.7596, -55.0000]
# b = [ 363.6550,  308.9218,  108.6540, 5293.0000]
# a = [  0.0000, 244.2279,  -0.7596, -55.0000]
# b = [ 363.6550,  308.9218,  108.6540, 5293.0000]
# a = [  0.0000, 244.1756,  -1.9270,   0.0000]
# b=[ 450.2442,  308.9454,  109.9388, 6761.0000]
a = [0.0000, 244.1756, -0.4107, 0.0000, 0.0000]
b = [450.2442, 308.9454, 109.0342, 6761.0000, 814.4700]  # yuanlaide
c = [b[i] - a[i] for i in range(len(a))]
transform1 = transforms.Compose([
    transforms.Normalize(mean=a,
                         std=c)
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform2 = transforms.Compose([
    transforms.Normalize(mean=[0], std=[483.6700])
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

max_g = 483.6700

train_data = TrainDataset('train', 'G:/downscale/gpm_0.25', 'G:/new_try/gpm_data', 'G:/new_try/tem_mean',
                          'G:/new_try/relative_hu_mean',
                          'G:/new_try/trmm_data', transform1=transform1, transform2=transform2)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print(len(train_data))

test_data = TrainDataset('val', 'G:/downscale/gpm_0.25', 'G:/new_try/gpm_data', 'G:/new_try/tem_mean',
                         'G:/new_try/relative_hu_mean',
                         'G:/new_try/trmm_data', transform1=transform1, transform2=transform2)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
img_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
print(len(test_data))

torch.manual_seed(123)


# 一种loss
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets, t):
        num = targets.size(0)
        smooth = 0.0001

        # probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m1 = torch.where(m1 >= t, 1, 0)
        m2 = targets.view(num, -1)
        m2 = torch.where(m2 >= t, 1, 0)
        intersection = (m1 * m2)

        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - 2 * score.sum() / num
        return score


class RAINLoss(nn.Module):
    def __init__(self):
        super(RAINLoss, self).__init__()

    def forward(self, pre, target):
        rain = torch.where(target >= 10, 1, 0)
        pre = pre.mul(rain)
        target = target.mul(rain)
        return nn.L1Loss()(pre, target)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


# criterion = nn.MSELoss()
criterion = nn.L1Loss()

# model = torch.load('G:/new_try\SRCNN/checkpoints_tp1/epoch_21.pth')

model = Generator(5, 2)

model = model.to(device)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)

print(model)
epoch_loss = []

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for epoch in range(epochs):
    model.train()
    train_loss = []
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_data) - len(train_data) % epochs), position=0) as t:
        t.set_description('epoch:{}/{}'.format(epoch, epochs - 1))

        for idx, (lr, hr) in enumerate(train_data_loader):
            # print(lr.shape, hr.shape)
            trmm = lr[:, 4:, :, :].to(device)
            lr = lr[:, :4, :, :]
            lr = lr.to(device)
            hr = hr.to(device).mul(max_g)
            lr_pre = lr[:, :1, :, :]

            # hr_gra = Get_gradient_nopadding().to(device)(hr)

            # lr = F.interpolate(lr, size=200, mode='bicubic', align_corners=True)  # 插值

            sr = model(lr, torch.cat((trmm, lr_pre), dim=1)).mul(max_g)
            # sr_gra = Get_gradient_nopadding().to(device)(sr)

            loss = criterion(sr, hr)  # +criterion(sr_gra, hr_gra)
            # + criterion(sr_gra* max_g, hr_gra* max_g)# + 0.1 * SoftDiceLoss()(sr * max_g, hr * max_g, 0.1)

            train_loss.append(loss.item())

            epoch_losses.update(criterion(sr, hr).item(), len(lr))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(lr))

    train_loss = np.average(train_loss)
    epoch_loss.append(train_loss)

    scheduler.step()

    print('train_loss: {}'.format(train_loss))
    torch.save(model, os.path.join(checkpoint_path, 'epoch_{}.pth'.format(epoch)))
    # np.save(os.path.join(checkpoint_path, 'loss'), epoch_loss)
    ## 测试效果
    model.eval()
    epoch_psnr = AverageMeter()
    pod = AverageMeter()
    csi = AverageMeter()
    far = AverageMeter()
    total_loss = []

    for data in test_data_loader:
        lr, hr = data

        trmm = lr[:, 4:, :, :].to(device)
        lr = lr[:, :4, :, :]
        lr = lr.to(device)
        hr = hr.to(device)
        lr_pre = lr[:, :1, :, :]

        # lr = F.interpolate(lr, size=200, mode='bicubic', align_corners=True)  # 插值

        with torch.no_grad():
            pre = model(lr, torch.cat((trmm, lr_pre), dim=1))
            loss = criterion(pre.detach().cpu() * max_g, hr.detach().cpu() * max_g)
            total_loss.append(loss)

            epoch_psnr.update(calc_psnr(pre, hr), len(lr))

            pre = pre[0, 0, :, :].detach().cpu().numpy()
            obs = hr[0, 0, :, :].detach().cpu().numpy()

            pre = pre * max_g
            obs = obs * max_g

            csi.update(CSI(obs=obs, pre=pre, threshold=0.1), len(lr))
            pod.update(POD(obs, pre, threshold=0.1), len(lr))
            far.update(FAR(obs, pre, threshold=0.1), len(lr))

    total_loss = np.average(total_loss)

    print('eval psnr: {:.2f}  val_loss: {}'.format(epoch_psnr.avg, total_loss))
    print('csi:{}  pod:{}  far:{}'.format(csi.avg, pod.avg, far.avg))
    print('\n')
