""" 3D brain age model"""
from box import Box
from torch import nn


def conv_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    )


class Model(nn.Module):
    def __init__(self, initialization="default", **kwargs):
        super(Model, self).__init__()
        self.initialization = initialization
        self.conv1 = conv_blk(1, 32)
        self.conv2 = conv_blk(32, 64)
        self.conv3 = conv_blk(64, 128)
        self.conv4 = conv_blk(128, 256)
        self.conv5 = conv_blk(256, 256)

        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        )

        self.drop = nn.Identity()  # nn.Dropout3d(p=0.5)

        self.output = nn.Conv3d(64, 1, kernel_size=1, stride=1)

        self.init_weights()

    def init_weights(self):
        if self.initialization == "custom":
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out",
                        nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.output.bias, 62.68)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.drop(x)
        x = self.output(x)
        return Box({"y_pred": x})


def get_arch(*args, **kwargs):
    return {"net": Model(**kwargs)}
