import einops
import torch
import torch.nn as nn
import torch.fft

class SRA(nn.Module):
    def __init__(self, dim : int, sr_ratio=2):
        super().__init__()
        self.dim = dim
        self.q = nn.Conv2d(256, dim, kernel_size=1, groups=dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(256, dim, kernel_size=3,padding=1,groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H*W
        x1 = x.reshape(B,N,C)
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = x_.reshape(B,-1,128)
            x2 = self.norm(x_)
        else:
            x2 = self.q(x).reshape(B,-1,128)
        return x2

class MH2SD(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim

        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.osr = SRA(dim, num_heads)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim)
        self.qkc = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.spe = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.bnc1 = nn.BatchNorm2d(dim)
        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=self.head_dim,
                               bias=False)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        qkv = self.qkc(self.act(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch * self.num_patch, self.inner_dim * 3)

        qkv = qkv.chunk(3, dim=-1)
        spe = self.spe(self.act(self.bn(x)))
        spe = self.avg_pool(spe)
        c =x
        q, k, v = map(lambda t: einops.rearrange(t, "b (h d) n -> b n h d", h=self.num_patch), qkv)
        qqkkvv= q = k = v
        qy = self.osr(qqkkvv)
        q= einops.rearrange(qy, "b n (h d) -> b h n d", h=self.num_heads)
        k=v=q
        spe = einops.rearrange(spe, "b (h d) n w -> b h (n w) d", h=self.num_heads)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        v_spe = torch.einsum("b h i j, b h j d -> b h i d", v, spe)
        v_spe = v_spe * self.scale
        v_spe1 = self.attn(v_spe)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_spe1)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        c = self.act(self.bnc1(self.local(c)))
        c = c.reshape(b, self.dim, self.num_patch, self.num_patch).reshape(b, n, -1)
        out = self.avgpool(out + c)

        return out


class FCE(nn.Module):

    def __init__(self, dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.depthwise_conv = nn.Sequential(

            nn.Conv2d(dim, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.GELU())

        self.squeeze_conv = nn.Sequential(

            nn.Conv2d(64, 16, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(16), nn.GELU())

        self.expand_conv = nn.Sequential(

            nn.Conv2d(16, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.depthwise_conv(x_reshape)
        out2 = self.squeeze_conv(out1)
        out3 = self.expand_conv(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)
        result = result+x
        return result


class LGPT_module(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), MH2SD(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FCE(dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mh, fce in self.layers:
            x = mh(x) + x
            x = fce(x) + x
        return x