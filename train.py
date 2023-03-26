import torch
import torch.nn as nn
from load_data_tg import TrainDataset
from model import FSRCNN
from generator_no_bn import Generator
from DBPN import Net
from RDN import RDN
# from generator_rdn import Generator
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
from fsrcnn import SRCNN
import time
from torch.nn.parallel import DistributedDataParallel as DDP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 16
epochs = 200
learning_rate = 0.0001  # 小一点?
localtime = time.localtime(time.time())
tm = str(localtime[0])+str(localtime[1])+str(localtime[2])+str(localtime[3])+str(localtime[4])+str(localtime[5])
checkpoint_path = os.path.join('G:/SQG/after3131/new_try/SRCNN/checkpoints_tp1/',tm)
#checkpoint_path = 'G:/new_try/SRCNN/checkpoints_tp1'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
#rank = int(os.environ("LOCAl_RANK"))
#print(rank)

# a = [  0.0000, 244.2279,  -0.7596, -55.0000]
# b = [ 363.6550,  308.9218,  108.6540, 5293.0000]
# a = [  0.0000, 244.2279,  -0.7596, -55.0000]
# b = [ 363.6550,  308.9218,  108.6540, 5293.0000]
a = [  0.0000, 244.1756,  -1.9270,   0.0000]
b=[ 450.2442,  308.9454,  109.9388, 6761.0000]
c = [b[i] - a[i] for i in range(len(a))]
transform1 = transforms.Compose([
    transforms.Normalize(mean=a,
                         std=c)
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform2 = transforms.Compose([
    transforms.Normalize(mean=[0], std=[438.6700])
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

max_g = 438.6700

train_data = TrainDataset('train', 'G:/downscale/gpm_0.25', 'G:/new_try/gpm_data', 'G:/new_try/tem_mean',
                          'G:/new_try/relative_hu_mean','G:/new_try/geopotential', transform1=transform1, transform2=transform2)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#print(train_data.shape)
print(len(train_data))

test_data = TrainDataset('val', 'G:/downscale/gpm_0.25', 'G:/new_try/gpm_data', 'G:/new_try/tem_mean',
                         'G:/new_try/relative_hu_mean',
                         'G:/new_try/geopotential', transform1=transform1, transform2=transform2)
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


# criterion = nn.MSELoss()
criterion = nn.L1Loss()

#model = torch.load('G:/new_try/SRCNN/checkpoints_tp1/epoch_21.pth')
model = Net(4, 16, 64)
#model = SRCNN(4)
model = RDN().to(device)
#model = model.to(device)
# if torch.cuda.device_count() > 1:
#     print("Use", torch.cuda.device_count(), 'gpus')
#     model = nn.DataParallel(model)

#print(model)
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
            print(lr.shape, hr.shape)
            #print(hr.shape)
            lr = lr.to(device)
            hr = hr.to(device)
            # hr_gra = Get_gradient_nopadding().to(device)(hr)

            lr = F.interpolate(lr, size=200, mode='bicubic',align_corners=True)  # 插值

            sr = model(lr)
            # sr_gra = Get_gradient_nopadding().to(device)(sr)
            print(sr.shape)
            loss = criterion(sr* max_g, hr* max_g) #+ criterion(sr_gra* max_g, hr_gra* max_g) + 0.1 * SoftDiceLoss()(sr * max_g, hr * max_g, 0.1)

            train_loss.append(loss.item())

            epoch_losses.update(loss.item(), len(lr))

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
    np.save(os.path.join(checkpoint_path, 'loss'), epoch_loss)
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
        lr = F.interpolate(lr, size=200, mode='bicubic', align_corners=True)  # 插值

        with torch.no_grad():
            pre = model(lr)
            loss = criterion(pre.detach().cpu(), hr.detach().cpu())
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
