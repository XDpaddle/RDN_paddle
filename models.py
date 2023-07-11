# import torch
# from torch import nn
import paddle
from paddle import nn

class DenseLayer(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return paddle.concat(x=[x, self.relu(self.conv(x))], axis=1)


class RDB(nn.Layer):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2D(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Layer):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2D(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2D(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        # self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        # for _ in range(self.D - 1):
        #     self.rdbs.append(RDB(self.G, self.G, self.C))


        self.rdbs = nn.Sequential(
                                RDB(self.G0, self.G, self.C),
                                *[RDB(self.G, self.G, self.C) for _ in range(self.D - 1)]
                                )
        
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2D(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2D(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2D(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2D(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2D(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(paddle.concat(x=local_features, axis=1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
