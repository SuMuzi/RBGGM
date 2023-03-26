import torch
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def unnormalized_show(img, std, mu):
    img = img * std + mu  # unnormalize
  #  npimg = img.numpy()
   # plt.figure()
  #  plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return  img

# a = torch.randn(3,2,2)
# b = unnormalized_show(a, 0.5 , 0.5)
# print(a, a.size())
# print(b, b.shape)
# print(torch.cuda.device_count())