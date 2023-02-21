import math

import torch
from torch import nn
from torchvision import models


def encoder1_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class Encoder1_2D(nn.Module):
    def __init__(
            self, out_size, slice_dim, use_position_encoding=False, in_channel=1, dropout=False,
            post_proc_conv=None, encoder_2d=None, post_position_encoding=None
    ):

        super().__init__()
        self.in_channel = in_channel
        if encoder_2d is None:
            encoder_blocks = [encoder1_blk(in_channel, 32), encoder1_blk(32, 64),
                              encoder1_blk(64, 128),
                              encoder1_blk(128, 256), encoder1_blk(256, 256)]
            self.encoder_2d = nn.Sequential(*encoder_blocks)
        else:
            self.encoder_2d = encoder_2d

        if slice_dim == 1:
            avg = nn.AvgPool2d([3, 2])
        elif slice_dim == 2:
            avg = nn.AvgPool2d([2, 2])
        elif slice_dim == 3:
            avg = nn.AvgPool2d([2, 3])
        else:
            raise Exception("Invalid slice dim")

        if post_proc_conv is None:
            self.post_proc_conv = nn.Sequential(
                nn.Conv2d(256, 64, 1, stride=1), nn.InstanceNorm2d(64), nn.ReLU(), avg
            )
        else:
            self.post_proc_conv = post_proc_conv

        if use_position_encoding:
            self.position_encoder = PositionEncodingUKBB(
                feature_dim=64, slice_dim=slice_dim, remove_slices=in_channel - 1
            )
        else:
            self.position_encoder = nn.Identity()

        if post_position_encoding is None:
            self.post_position_encoding = nn.Sequential(
                nn.Dropout(p=0.5) if dropout else nn.Identity(),
                nn.Conv2d(64, out_size, 1)
            )
        else:
            self.post_position_encoding = post_position_encoding

    def forward(self, x):
        # breakpoint()

        return self.post_position_encoding(
            self.position_encoder(
                self.post_proc_conv(
                    self.encoder_2d(x)
                )
            )
        )




class MeanPool(nn.Module):
    def forward(self, X):
        return X.mean(dim=1, keepdim=True), None


class MaxPool(nn.Module):
    def forward(self, X):
        return X.max(dim=1, keepdim=True)[0], None


class PooledAttention(nn.Module):
    def __init__(self, input_dim, dim_v, dim_k, num_heads, ln=False):
        super(PooledAttention, self).__init__()
        self.S = nn.Parameter(torch.zeros(1, dim_k))
        nn.init.xavier_uniform_(self.S)

        # no need to transform the query vector
        # self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(input_dim, dim_k)
        self.fc_v = nn.Linear(input_dim, dim_v)

        self.dim_v = dim_v
        self.dim_k = dim_k
        self.num_heads = num_heads

        if ln:
            self.ln0 = nn.LayerNorm(dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, X):
        B, C, H = X.shape

        Q = self.S.repeat(X.size(0), 1, 1)

        K = self.fc_k(X.reshape(-1, H)).reshape(B, C, self.dim_k)
        V = self.fc_v(X.reshape(-1, H)).reshape(B, C, self.dim_v)
        dim_split = self.dim_v // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        O = torch.cat(A.bmm(V_).split(B, 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = self.fc_o(O)
        return O, A

    def get_attention(self, X):
        B, C, H = X.shape

        Q = self.S.repeat(X.size(0), 1, 1)

        K = self.fc_k(X.reshape(-1, H)).reshape(B, C, self.dim_k)
        V = self.fc_v(X.reshape(-1, H)).reshape(B, C, self.dim_v)
        dim_split = self.dim_v // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        return A


class PositionEncodingUKBB(nn.Module):
    def __init__(self, feature_dim, slice_dim, remove_slices=0):
        super(PositionEncodingUKBB, self).__init__()
        if slice_dim == 1:
            sizes = [1, feature_dim, 91 - remove_slices, 1, 1]

        if slice_dim == 2:
            sizes = [1, feature_dim, 1, 109 - remove_slices, 1]

        if slice_dim == 3:
            sizes = [1, feature_dim, 1, 1, 91 - remove_slices]

        self.position_encoding = nn.Parameter(torch.zeros(*sizes))
        nn.init.kaiming_normal_(self.position_encoding, mode="fan_out", nonlinearity="linear")
        self.ln = nn.LayerNorm(sizes[1:])
        self.slice_dim = slice_dim

    def extra_repr(self) -> str:
        return 'sizes={}'.format(self.position_encoding.shape)

    def forward(self, x):
        # we need to concatenate, add, and split again
        N = x.shape[0]
        if self.slice_dim == 1:
            B = N // 91
            encoding = torch.cat(
                [i.unsqueeze(2) for i in torch.split(x, B, dim=0)], dim=2
            )
            # breakpoint()
            # import ipdb;ipdb.set_trace()
            encoding = self.ln(encoding + self.position_encoding)
            H = encoding.shape[2]
            return torch.cat([encoding[:, :, i, :, :] for i in range(H)], dim=0)

        if self.slice_dim == 2:
            B = N // 109
            encoding = torch.cat(
                [i.unsqueeze(3) for i in torch.split(x, B, dim=0)], dim=3
            )
            encoding = self.ln(encoding + self.position_encoding)
            W = encoding.shape[3]
            return torch.cat([encoding[:, :, :, i, :] for i in range(W)], dim=0)

        if self.slice_dim == 3:
            B = N // 91
            encoding = torch.cat(
                [i.unsqueeze(4) for i in torch.split(x, B, dim=0)], dim=4
            )
            encoding = self.ln(encoding + self.position_encoding)
            D = encoding.shape[4]
            return torch.cat([encoding[:, :, :, :, i] for i in range(D)], dim=0)


class ResNetEncoder(nn.Module):
    def __init__(
            self, out_size, slice_dim, use_position_encoding=False, in_channel=1, pretrained=False,
            resnet_module="resnet18", **kwargs,
    ):
        super().__init__()
        self.in_channel = in_channel

        self.encoder_2d, self.encoding_dim = self.get_encoder2D(resnet_module, pretrained)

        self.post_proc = nn.Sequential(nn.Linear(self.encoding_dim, out_size))

        if use_position_encoding:
            self.position_encoder = PositionEncodingUKBB(
                feature_dim=out_size, slice_dim=slice_dim, remove_slices=in_channel - 1
            )
        else:
            self.position_encoder = nn.Identity()

    def get_encoder2D(self, module, pretrained):
        if module == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            output_dim = 512
        elif module == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            output_dim = 1024
        elif module == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            output_dim = 2048
        else:
            raise NotImplementedError(f"{module} not implemented")
        del model.fc
        return model, output_dim

    def forward(self, x):
        if x.size()[1] == 1:
            # if we get only one channel, we need to replicate
            x = torch.cat([x, x, x], dim=1)
        # breakpoint()
        # this will BX64,and we will add two more dimension so it can be consistent
        out = self.post_proc(self.get_encoder_2d_embeddings(x))
        out = out.unsqueeze(2).unsqueeze(3)
        out = self.position_encoder(out)
        return out 

    def get_encoder_2d_embeddings(self, x):
        x = self.encoder_2d.conv1(x)
        x = self.encoder_2d.bn1(x)
        x = self.encoder_2d.relu(x)
        x = self.encoder_2d.maxpool(x)

        x = self.encoder_2d.layer1(x)
        x = self.encoder_2d.layer2(x)
        x = self.encoder_2d.layer3(x)
        x = self.encoder_2d.layer4(x)
        x = self.encoder_2d.avgpool(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x
