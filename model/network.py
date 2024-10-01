import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from Transformer import LGPT_module
from Embeddings import PatchEmbeddings, PositionalEmbeddings
from mmcv.cnn.bricks import ConvModule, build_activation_layer

class DDC_module(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=2,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1), )
        self.proj1 = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        x.cuda()
        B, C, H, W = x.shape
        x1 = self.pool(x)
        scale1 = self.proj(x1)
        scale3 = torch.cat((scale1,x1),dim=1)
        scale4 = scale3.reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale4, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj1(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)



class MultiScaleDWConv(nn.Module):
    # def __init__(self, dim, scale=(1, 3, 5, 7)):#原始
    def __init__(self, dim, scale=(1, 3,5,7)):#消融实验
        super().__init__()

        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x






class MsEF_module(nn.Module):
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, in_features//6, kernel_size=1, bias=False),#
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(in_features//6),
        )
        self.dwconv = MultiScaleDWConv(in_features//6)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(in_features//6)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_features//6, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class LRDTN(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C,
                 patch_size, image_size: int =121, emb_dim: int = 128, num_layers: int = 1,
                 num_heads: int =4, head_dim = 64, hidden_dim: int = 128, attn_drop : int = 0, sr_ration:int = 1,dim1=128,
                 act_cfg=dict(type='GELU')):
        super(LRDTN, self).__init__()
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = HSI_Data_Shape_C
        self.image_size = image_size
        self.num_patches = 121
        self.attn_drop = attn_drop
        self.sr_ration = sr_ration
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = HSI_Data_Shape_C
        kernel_size=3
        num_groups=2
        drop=0
        self.dim1=dim1
        self.relu = nn.ReLU()

        """branch 1"""
        self.DDC = DDC_module(dim=self.band, kernel_size=kernel_size, num_groups=num_groups)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )


        """branch 2"""

        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.LGPT = LGPT_module(dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
                                        head_dim=head_dim, num_patch=self.num_patch, patch_size=patch_size)

        self.MsEF = MsEF_module(in_features=self.band,
                       act_cfg = act_cfg,
                       drop = attn_drop, )

        self.Dconv21 = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(emb_dim),
        )
        self.drop = nn.Dropout(drop)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.finally_fc_classification = nn.Linear(128, self.classes)

    def forward(self,pixelX,patchX):
        """------------------------branch 1------------------------"""
        pixelX = pixelX.cuda()
        x11 = self.DDC(pixelX)
        x12 = x11 + pixelX
        x13 = self.MsEF(x12)
        x14 = self.conv11(x13+pixelX)
        output_1 = self.global_pooling(x14)

        """------------------------branch 2------------------------"""

        patchX=patchX.cuda()
        x21 = self.patch_embeddings(patchX)
        x22 = self.pos_embeddings(x21)
        x23 = self.LGPT(x22)
        x24 = torch.mean(x23, dim=1)
        output_2 = x24.unsqueeze(-1).unsqueeze(-1)

        """------------------------fusion------------------------"""

        output3 = output_1 + output_2
        output4 = self.Dconv21(output3)
        output5 = self.drop(output4)
        output6 = torch.squeeze(output5,dim=(2,3))
        output7 = self.finally_fc_classification(output6)
        output = F.softmax(output7, dim=1)

        return output, output,output7

