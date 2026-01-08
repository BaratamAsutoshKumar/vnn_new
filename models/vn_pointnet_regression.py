import torch
import torch.nn as nn
from models.vn_layers import VNLinear, VNBatchNorm, VNLeakyReLU


class VNPointNetRegressor(nn.Module):
    """
    VNN PointNet-style regressor

    Input:
        x : (B, C, 3, N)   (C=1 for raw points)

    Output:
        y : (B, 1, 3, N)   predicted transformed points
    """

    def __init__(self):
        super().__init__()

        self.conv1 = VNLinear(1, 64)
        self.bn1 = VNBatchNorm(64, dim=3)
        self.act1 = VNLeakyReLU(64)

        self.conv2 = VNLinear(64, 128)
        self.bn2 = VNBatchNorm(128, dim=3)
        self.act2 = VNLeakyReLU(128)

        self.conv3 = VNLinear(128, 256)
        self.bn3 = VNBatchNorm(256, dim=3)
        self.act3 = VNLeakyReLU(256)

        self.conv4 = VNLinear(256, 128)
        self.bn4 = VNBatchNorm(128, dim=3)
        self.act4 = VNLeakyReLU(128)

        self.conv5 = VNLinear(128, 1)   # back to 1 vector channel

    def forward(self, x):
        """
        x: (B, C, 3, N)
        """

        x = self.act1(self.bn1(self.conv1(x)))   # (B,64,3,N)
        x = self.act2(self.bn2(self.conv2(x)))   # (B,128,3,N)
        x = self.act3(self.bn3(self.conv3(x)))   # (B,256,3,N)
        x = self.act4(self.bn4(self.conv4(x)))   # (B,128,3,N)
        x = self.conv5(x)                        # (B,1,3,N)

        return x
