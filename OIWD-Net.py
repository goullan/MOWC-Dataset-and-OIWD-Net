# https://github.com/rishikksh20/ResMLP-pytorch

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from torchvision.transforms import Resize

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLPblock(nn.Module):

    def __init__(self, dim, num_patch, mlp_dim, dropout=0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        # x=x.view(x.size(0),-1)
        return x

class ResMLP(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))
        self.affine = Aff(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)
        self.sa = SpatialAttention()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(401408, 32)  # 114688  200704
        # self.linear2 = nn.Linear(300, 32)
        self.drop = nn.Dropout(0.5)

    def make_post(self, ch_in):
        layers = [

            nn.BatchNorm2d(ch_in, momentum=0.99),
            nn.LeakyReLU(0.3)
            # nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    # 定义核心模块 空间可分离卷积 深度可分离卷积 最大池化  作用:降低计算量
    def make_layers(self, ch_in, ch_out):
        layers = [
            # ch_in=32,  (3,32)
            nn.Conv2d(1, ch_in, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=0,
                      dilation=(1, 1)) if ch_in == 32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), stride=(1, 1),
                                                                     bias=False, padding=0, dilation=(1, 1)),
            self.make_post(ch_in),
            # DW
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3), padding=(0, 1),
                      bias=False, dilation=(1, 1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # DW
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                      bias=False, dilation=(1, 1)),
            self.make_post(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, dilation=(1, 1)),
            self.make_post(ch_out)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        x = self.affine(x)
        # x = x.unsqueeze(1)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        x = torch.reshape(x, [8, 224, 224])
        x = x.unsqueeze(1)
        x = self.block1(x)
        # x = self.drop2(x)
        # x = self.eca1(x) * x
        x = self.ca1(x) * x
        x = self.sa(x) * x
        # x = self.sig(x)
        # x = self.se1(x)

        x = self.block2(x)
        # x = self.drop2(x)
        # x = self.eca2(x) * x
        x = self.ca2(x) * x
        x = self.sa(x) * x

        # x = self.block3(x)
        # # x = self.drop2(x)
        # # x = self.eca2(x) * x
        # x = self.ca3(x) * x
        # x = self.sa(x) * x

        x = self.flatten(x)
        x = self.drop(x)
        # x = self.sig(x)
        x = self.linear1(x)
        # x = self.drop(x)
        # x = self.sig(x)
        # x = self.linear2(x)
        return x


if __name__ == "__main__":
    img = torch.ones([1, 2, 350, 350])

    model = ResMLP(in_channels=2, image_size=350, patch_size=14, num_classes=224*224,
                   dim=384, depth=14, mlp_dim=384 * 4)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)
    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]