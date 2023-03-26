import torch
import torch.nn as nn
from load_data_tg import TrainDataset
from DRN_re import DRN_rev
from generator_old import Generator

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 16
epochs = 50
learning_rate = 0.0001
checkpoint_path = 'G:/new_try/SRCNN/checkpoints_tp'
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

# transform1 = transforms.Normalize(mean=[0],
#                                   std=[401.3700])
#
# transform2 = transforms.Normalize(mean=[0],
#                                   std=[266.6850])

transform1 = transforms.Compose([
    transforms.Normalize(mean=[0.0000, 252.3060, 4.0818, 0.0000],
                         std=[814.4700, 319.3127 - 252.3060, 110.3622 - 4.0818, 6761.0000])
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform2 = transforms.Compose([
    transforms.Normalize(mean=[0], std=[438.6700])
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

max_g = 438.6700

train_data = TrainDataset('train','G:/downscale/gpm', 'G:/new_try/gpm_data', 'G:/new_try/tem', 'G:/new_try/relative_hu','G:/new_try/geopotential', transform1=transform1, transform2=transform2)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print(len(train_data))

test_data = TrainDataset('val','G:/downscale/gpm', 'G:/new_try/gpm_test', 'G:/new_try/tem_test', 'G:/new_try/r_h_test',
                         'G:/new_try/geopotential_test', transform1=transform1, transform2=transform2)
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


criterion = nn.MSELoss()
model = torch.load('G:/new_try/SRCNN/直接改为了gpm_down到gpm/epoch_49.pth')
model_re = DRN_rev().to(device)
# model = RDN().to(device)

# if torch.cuda.device_count() > 1:
#     print("Use", torch.cuda.device_count(), 'gpus')
#     model = nn.DataParallel(model)

print(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.001)
optimizer_re = optim.Adam(model_re.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    model.train()
    model_re.train()
    train_loss = []
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_data) - len(train_data) % epochs), position=0) as t:
        t.set_description('epoch:{}/{}'.format(epoch, epochs - 1))

        for idx, (lr, hr) in enumerate(train_data_loader):
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            lr_re = model_re(sr)

            # compute primary loss
            loss_primary = criterion(sr, hr)

            # compute dual loss
            loss_re = criterion(lr[:, :1, :, :], lr_re)

            # compute total loss
            loss = loss_primary + 0.1*loss_re

            train_loss.append(loss.item())

            epoch_losses.update(loss.item(), len(lr))

            optimizer.zero_grad()
            optimizer_re.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_re.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(lr))

    train_loss = np.average(train_loss)

    # scheduler.step()

    print('train_loss: {}'.format(train_loss))
    torch.save(model, os.path.join(checkpoint_path, 'epoch_{}.pth'.format(epoch)))

    ## 测试效果
    model.eval()
    epoch_psnr = AverageMeter()
    pod = AverageMeter()
    csi = AverageMeter()
    far = AverageMeter()
    total_loss = []

    for data in test_data_loader:
        lr, hr = data

        lr = lr.to(device)
        hr = hr.to(device)

        # lr = F.interpolate(hr, size=80, mode='bicubic', align_corners=True)

        with torch.no_grad():
            pre = model(lr)
            loss = criterion(pre.detach().cpu(), hr.detach().cpu())
            total_loss.append(loss)

            epoch_psnr.update(calc_psnr(pre, hr), len(lr))

            pre = pre[:, 0, :, :].detach().cpu().numpy()
            obs = hr[:, 0, :, :].detach().cpu().numpy()

            # max_t = torch.from_numpy(max_t).type_as(pre).to(pre.device) if torch.is_tensor(pre) else max_t
            # max_g = torch.from_numpy(max_g).type_as(pre).to(pre.device) if torch.is_tensor(pre) else max_g

            pre = pre * max_g
            obs = obs * max_g
            # pre = pre
            # obs = obs

            csi.update(CSI(obs=obs, pre=pre, threshold=0.1), len(lr))
            pod.update(POD(obs, pre, threshold=0.1), len(lr))
            far.update(FAR(obs, pre, threshold=0.1), len(lr))

    total_loss = np.average(total_loss)

    print('eval psnr: {:.2f}  val_loss: {}'.format(epoch_psnr.avg, total_loss))
    print('csi:{}  pod:{}  far:{}'.format(csi.avg, pod.avg, far.avg))
    print('\n')
