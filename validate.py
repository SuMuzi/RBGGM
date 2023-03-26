import torch
import torch.nn as nn
from load_data_monthpre import TrainDataset
# from model import FSRCNN
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
import matplotlib.pyplot as plt
import scipy.ndimage
from Srsp import Get_gradient_nopadding
from pre_plot import PLOT_pre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_g = 483.6700

# a = [0.0000, 244.1756, -1.9270, 0.0000]
# b = [450.2442, 308.9454, 109.9388, 6761.0000]
a = [0.0000, 244.1756, -0.4107, 0.0000]
b = [450.2442, 308.9454, 109.0342, 6761.0000]
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
transform3 = transforms.Compose([
    transforms.Normalize(mean=[0], std=[3.2556])
])

test_data = TrainDataset('test', 'G:/new_try/gpm_data', 'G:/new_try/gpm_data', 'G:/new_try/tem_mean',#G:/new_try/trmm_data
                         'G:/new_try/relative_hu_mean',
                         'G:/new_try/trmm_data', transform1=transform1,
                         transform2=transform2, transform3=transform3)

img_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# model = torch.load(r'D:\新建文件夹\month，pre， 64×3， feature later\epoch_199.pth')
model = torch.load(r'G:\new_try\X8\srcnn/epoch_103.pth')


# model = torch.load(r'G:\new_try\SRCNN\7000-4000\新建文件夹\GRA, 4LOSS, NEW-ATT/epoch_159.pth')
# model = torch.load(r'G:\new_try\SRCNN\7000-4000\新建文件夹\generator，1+3，mae，e_d V3/epoch_29.pth')
# model = torch.load(r'G:\new_try\SRCNN\7000-4000\新建文件夹\GRA, 4LOSS\gra， 4loss_2/epoch_99.pth')


# checkpoints_tp/
def plot_con(pre1, pre2, tem, hm, z, sr, inter, grad):
    x = np.linspace(100, 120, 200)
    y = np.linspace(20, 40, 200)
    x_l = np.linspace(100, 120, 80)
    y_l = np.linspace(20, 40, 80)
    # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    X, Y = np.meshgrid(x, y)
    X_l, Y_l = np.meshgrid(x_l, y_l)
    x_2 = np.linspace(60, 140, 80)
    y_2 = np.linspace(60, 140, 80)
    X_2, Y_2 = np.meshgrid(x_2, y_2)

    # 填充等高线
    # plt.subplot(1, 2, 1)
    # plt.contourf(X, Y, pre1, cmap=plt.cm.get_cmap('Spectral_r'))
    # plt.subplot(1, 2, 2)
    # plt.contourf(X, Y, pre2, cmap=plt.cm.get_cmap('Spectral_r'))

    plt.subplot(2, 4, 1)
    plt.contourf(X, Y, pre1, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 50, 400))  #
    plt.title('pre_orgin')
    plt.subplot(2, 4, 2)
    plt.contourf(X_l, Y_l, pre2, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 50, 400))
    plt.title('pre_0.25')
    plt.subplot(2, 4, 3)
    plt.contourf(X_2, Y_2, tem, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 50, 400))
    plt.title('hr')
    plt.subplot(2, 4, 4)
    plt.contourf(X_l, Y_l, hm, cmap=plt.cm.get_cmap('Spectral_r'))
    plt.title('h_r')
    plt.subplot(2, 4, 5)
    plt.contourf(X, Y, z, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 30, 400))
    plt.title('sr_gra')
    plt.subplot(2, 4, 6)
    plt.contourf(X, Y, sr, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 50, 400))
    plt.title('output')
    plt.subplot(2, 4, 7)
    plt.contourf(X_2, Y_2, inter, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 50, 400))  #
    plt.title('sr')
    plt.subplot(2, 4, 8)
    plt.contourf(X, Y, grad, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 30, 400))
    plt.title('hr_grad')

    # plt.savefig(r'C:\Users\hello\Desktop\1.png', dpi=300)
    # 显示图表
    plt.show()


for i, (lr, month_pre, hr) in enumerate(img_data_loader):
    lr = F.interpolate(lr, size=200, mode='bicubic', align_corners=True)
    lr = lr.to(device)
    # print(lr.shape)

    lr_pre = lr[:, :1, :, :]
    hr = hr.to(device)
    mon_pre = month_pre.to(device)

    # sr = model(lr, mon_pre, lr_pre)[1].clamp(0, 1)
    sr = model(lr).clamp(0, 1)
    # sr_gra = Get_gradient_nopadding().to(device)(sr) * max_g
    # hr_gra = Get_gradient_nopadding().to(device)(hr.to(device)) * max_g
    # bicubic = nn.functional.interpolate(hr, scale_factor=0.4, mode='bicubic').clamp(0,1)

    hr = hr.cpu().numpy()[0, 0, :, :]
    lr = lr.cpu().numpy()[0, :, :, :]
    sr = sr.cpu().detach().numpy()[0, 0, :, :]

    sr = sr * max_g
    hr = hr * max_g
    lr[0, :, :] = (lr[0, :, :]) * 450.2442

    # np.save(r'G:\result_X8\gpm\{}'.format(i), hr)
    np.save(r'G:\result_X8\srcnn\{}'.format(i), sr)
    # np.save(r'G:\result_X8\gpm_0.4\{}'.format(i), lr)
    print('sr',sr.shape)
    # hr_do = scipy.ndimage.zoom(hr, 0.4, order=1)
    # f_sr = scipy.ndimage.zoom(lr[0, :, :], 2.5, order=1)
    # lr_gra = torch.from_numpy(hr_do).unsqueeze(0).unsqueeze(1)
    # lr_gra = Get_gradient_nopadding().to(device)(lr_gra.to(device))
    # lr_gra = lr_gra.cpu().numpy()

    print('sr:', np.min(sr), np.max(sr), 'hr:', np.min(hr), np.max(hr), 'lr:', np.max(lr[0, :, :]), np.min(lr[0, :, :]))
    # plot_con(hr, lr[0, :, :], lr[1, :, :], lr[2, :, :], lr[3, :, :], sr)bicubic.cpu().numpy()[0, 0, :, :]*max_g
    # plot_con(hr, trmm.cpu()[0,0,:,:]*814.4700, hr[60:140,60:140], lr[2, :, :], sr_gra[0, 0, :, :].cpu().detach().numpy(), sr, sr[60:140,60:140], hr_gra.cpu().numpy()[0, 0, :, :])

    # PLOT_pre(hr, sr, 'hr', 'g-rcan')


    #
    # x = np.linspace(100, 120, 200)
    # y = np.linspace(20, 40, 200)
    # X, Y = np.meshgrid(x, y)
    #
    # x1 = np.linspace(100, 120, 80)
    # y1 = np.linspace(20, 40, 80)
    # X1, Y1 = np.meshgrid(x1, y1)
    # plt.subplot(1, 2, 1)
    # plt.contourf(X1, Y1, lr[0, :, :], cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 80, 400))  #
    # plt.title('hr')
    # plt.subplot(1, 2, 2)
    # plt.contourf(X, Y, sr, cmap=plt.cm.get_cmap('Spectral_r'), levels=np.linspace(0, 80, 400))
    # plt.title('grcan')
    # plt.show()
