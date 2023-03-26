from torch import nn
import math


# class SRCNN(nn.Module):
#     def __init__(self, in_channels=1, out_channels = 1):
#         super(SRCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
#         self.upscale = Upsampler()
#         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=5 // 2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.upscale(x)
#         x = self.conv3(x)
#         return x


class FSRCNN(nn.Module):
    def __init__(self, scale_factor=5, num_channels=4, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                             output_padding=scale_factor - 1)
        self.end = nn.Sequential(
            nn.Conv2d(num_channels, 1, 2, 2, 0),
            nn.PReLU(1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        x = self.end(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, n_feats=32):
        super(Upsampler, self).__init__()
        self.conv = nn.Conv2d(n_feats, 25 * n_feats, 1, stride=1, padding=0)
        self.shuffler = nn.PixelShuffle(5)
        self.con_2 = nn.Conv2d(n_feats, 1, 2, 2, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.con_2(self.shuffler(self.conv(x))))

# class Upsampler(nn.Module):
#     def __init__(self, n_feats=32):
#         super(Upsampler, self).__init__()
#         self.conv = nn.Conv2d(n_feats, 25 * n_feats, 2, stride=2, padding=0)
#         self.shuffler = nn.PixelShuffle(5)
#         self.relu = nn.ReLU(inplace=True)


