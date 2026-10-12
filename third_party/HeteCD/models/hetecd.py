import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from models.base_block import *
from torchvision.models import resnet18
import torch.nn.functional as F
import warnings
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
import pdb
from scipy.io import savemat


class Backbone(nn.Module):
    def __init__(self, patch_size=7, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        #----------------------------------#
        #   Transformer模块，共有四个部分
        #----------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]     
        #----------------------------------#
        #   block1
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   512, 512, 3 => 128, 128, 32 => 16384, 32
        #-----------------------------------------------#
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   16384, 32 => 16384, 32
        #-----------------------------------------------#
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])
        
        #----------------------------------#
        #   block2
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   128, 128, 32 => 64, 64, 64 => 4096, 64
        #-----------------------------------------------#
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   4096, 64 => 4096, 64
        #-----------------------------------------------#
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])
        #----------------------------------#
        #   block3
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   64, 64, 64 => 32, 32, 160 => 1024, 160
        #-----------------------------------------------#
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   1024, 160 => 1024, 160
        #-----------------------------------------------#
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        #----------------------------------#
        #   block4
        #----------------------------------#
        #-----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   32, 32, 160 => 16, 16, 256 => 256, 256
        #-----------------------------------------------#
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        #-----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   256, 256 => 256, 256
        #-----------------------------------------------#
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        #----------------------------------#
        #   block1
        #----------------------------------#
        x, H, W = self.patch_embed1.forward(x)

        for i, blk in enumerate(self.block1):
            x = blk.forward(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #----------------------------------#
        #   block2
        #----------------------------------#
        x, H, W = self.patch_embed2.forward(x)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #----------------------------------#
        #   block3
        #----------------------------------#
        x, H, W = self.patch_embed3.forward(x)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #----------------------------------#
        #   block4
        #----------------------------------#
        x, H, W = self.patch_embed4.forward(x)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self,patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim        = dim
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(dim)
        self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop  = nn.Dropout(attn_drop)
        
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class DWCPatchEmbed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride, 
            act_layer=act_layer,
        )

    def forward(self, x):

        x = self.patch_conv(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP_PRED(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class change_region_enhance(nn.Module):
    def __init__(self,in_channels,gamma=2,b=1):
        super(change_region_enhance, self).__init__()
        k=int(abs((math.log(in_channels,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )
        self.conv_restore=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,padding=0,bias=False)
        )
    def forward(self,x_list):
        x = x_list[1]
        bcd_map = x_list[0]
        b, c, h, w = x.shape
        bcd_map = resize(bcd_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        bcd_map = bcd_map.expand_as(x)
        # atten=self.pool(x)
        # atten=atten.view(x.size(0),1,x.size(1))
        # atten=self.conv(atten)
        # atten=atten.view(x.size(0),x.size(1),1,1)
        # bcd_map = bcd_map * atten
        x = x*bcd_map
        x = self.conv_restore(x)
        return x   
    

class cross_fuse(nn.Module):
    def __init__(self, in_channels):
        super(cross_fuse, self).__init__()
        self.g_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True, groups = in_channels//2),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.dropout = None
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b,c,h,w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w )
        tensor2 = tensor2.view(b, c, h*w )
        cross_x = torch.cat((tensor1, tensor2), dim=2)
        cross_x = cross_x.view(b, c*2, h,w)
        cross_x = self.g_conv(cross_x)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)
        return cross_x

# class cross_fuse_3d(nn.Module):
#     def __init__(self, in_channels):
#         super(cross_fuse_3d, self).__init__()
#         self.conv3d = nn.Sequential(
#             nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
#             nn.ReLU(),

#             nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
#             nn.ReLU(),

#             nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
#             nn.ReLU(),

#             nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
#             nn.ReLU(),
            

#         )
#         # self.fuse_conv = nn.Sequential(
#         #     nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
#         #     nn.ReLU(),
#         #     nn.BatchNorm2d(in_channels//2),
#         # )
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.dropout = nn.Dropout2d(0.1)
#     def forward(self, x):
#         tensor1 = x[0]
#         tensor2 = x[1]
#         b,c,h,w = tensor1.shape
#         tensor1 = tensor1.view(b, c, h*w )
#         tensor2 = tensor2.view(b, c, h*w )
#         cross_x = torch.cat((tensor1, tensor2), dim=2)
#         cross_x = cross_x.view(b, c*2, h,w)
#         cross_x = cross_x.unsqueeze(1)
#         cross_x = self.conv3d(cross_x)
#         cross_x = cross_x.squeeze(1)
#         cross_x = self.fuse_conv(cross_x)
#         cross_x = self.dropout(cross_x)
#         # # print(cross_x.shape)

#         return cross_x
class cross_fuse_3d(nn.Module):
    def __init__(self, in_channels):
        super(cross_fuse_3d, self).__init__()
        self.conv3d1 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))                 
        self.conv3d2 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,5,5], stride=1, padding=(1, 2, 2), bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))      
        self.conv3d3 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,7,7], stride=1, padding=(1, 3, 3), bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))     

        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(in_channels//2),
        # )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
        )
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b,c,h,w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w )
        tensor2 = tensor2.view(b, c, h*w )
        cross_x = torch.cat((tensor1, tensor2), dim=2)
        cross_x = cross_x.view(b, c*2, h,w)
        cross_x = cross_x.unsqueeze(1)
        cross_x1 = self.conv3d1(cross_x)
        cross_x1 = cross_x1.squeeze(1)
        cross_x2 = self.conv3d2(cross_x)
        cross_x2 = cross_x2.squeeze(1)
        cross_x3 = self.conv3d3(cross_x)
        cross_x3 = cross_x3.squeeze(1)
        cross_x = torch.cat((cross_x1, cross_x2, cross_x3), dim=1)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)
        cross_x = self.fuse_conv1(cross_x)
        cross_x = self.dropout(cross_x)
        # cross_x = torch.cat((tensor1, tensor2), dim=1)
        # # print(cross_x.shape)
        # cross_x = self.fuse_conv(cross_x)
        # cross_x = self.dropout(cross_x)
        # cross_x = self.fuse_conv1(cross_x)
        # cross_x = self.dropout(cross_x)
        return cross_x

class diff(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(diff, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv2d_cd = nn.Sequential(
        #     Conv2d_cd(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False, theta= 0.7),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(out_channels//2),
        # )
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels//2),
        )
        # self.channel_atten = channel_atten(out_channels)
        self.out_conv = nn.Sequential(            
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        #feat1 = self.conv2d_cd(x)
        feat2 = self.conv2d(x)
        #x = torch.cat([feat1, feat2], dim=1)
        #x = self.channel_atten(x)

        out = self.out_conv(feat2)
        return out

#Intermediate prediction module
def make_prediction(in_channels, out_channels, sigmoid=False):
    if sigmoid:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
# def change_region_enhance(bcd_map, x):
#     b, c, h, w = x.shape
#     bcd_map = resize(bcd_map, size=x.size()[2:], mode='bilinear', align_corners=False)
#     bcd_map = bcd_map.expand(b, c, h, w)
#     x = x * bcd_map+x
#     return x
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class SemanticHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SemanticHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x,_c

class semantic_context_enhance(nn.Module):
    def __init__(self):
        super(semantic_context_enhance, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_s = x[0]
        x_c = x[1]
        avgout = torch.mean(x_s, dim=1, keepdim=True)
        maxout, _ = torch.max(x_s, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = x_c*out
        return out
    

class CD_3D_Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=9, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16]):
        super(CD_3D_Decoder, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP_PRED(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP_PRED(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP_PRED(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP_PRED(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
    

        #convolutional Difference Modules
        self.cross_fuse_c4 = cross_fuse_3d_atten(in_channels=2*self.embedding_dim)
        self.cross_fuse_c3 = cross_fuse_3d_atten(in_channels=2*self.embedding_dim)
        self.cross_fuse_c2 = cross_fuse_3d_atten(in_channels=2*self.embedding_dim)
        self.cross_fuse_c1 = cross_fuse_3d_atten(in_channels=2*self.embedding_dim)


        #taking outputs from middle of the encoder
        self.sce = semantic_context_enhance()
        self.in_channels = [64, 128, 320, 512]

        self.make_pred_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=2,sigmoid = False)
        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        self.linear_fuse_bcd = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.dense_2x_bcd   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.dense_1x_bcd   = nn.Sequential( ResidualBlock(self.embedding_dim))  
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Softmax() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape
        outputs = []
        
        
        
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        diff_feat_c4 = self.cross_fuse_c4([_c4_1, _c4_2])
        diff_feat_c4_up = resize(diff_feat_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        


        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        diff_feat_c3 = self.cross_fuse_c3([_c3_1, _c3_2]) + F.interpolate(diff_feat_c4, scale_factor=2, mode="bilinear")
        # diff_feat_c3 = self.sce([resize(diff_s, size=diff_feat_c3.size()[2:], mode='bilinear', align_corners=False),diff_feat_c3])
        diff_feat_c3_up = resize(diff_feat_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
                # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.cross_fuse_c2([_c2_1, _c2_2]) + F.interpolate(diff_feat_c3, scale_factor=2, mode="bilinear")
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.cross_fuse_c1([_c1_1, _c1_2]) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

        #Linear Fusion of difference image from all scales
        _c_bcd = self.linear_fuse_bcd(torch.cat((diff_feat_c4_up, diff_feat_c3_up,_c2_up,_c1), dim=1))
        x = self.dense_2x(_c_bcd)
        x = self.dense_1x(x)
        p_bcd = self.make_pred_bcd(x)
        # print('p_bcd', p_bcd.shape)
        # print(c1_1.size(),c2_1.size(),c3_1.size(),c4_1.size())

        

        outputs.append(p_bcd)  
        # if self.output_softmax:
        #     temp = outputs
        #     outputs = []
        #     for pred in temp:
        #         outputs.append(self.active(pred))

        return outputs
class CD_Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=9, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16]):
        super(CD_Decoder, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP_PRED(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP_PRED(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP_PRED(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP_PRED(input_dim=c1_in_channels, embed_dim=self.embedding_dim)
    

        #convolutional Difference Modules
        self.cross_fuse_cs = cross_fuse_3d(in_channels=2*self.embedding_dim)
        self.cross_fuse_c4 = cross_fuse_3d(in_channels=2*self.embedding_dim)
        self.cross_fuse_c3 = cross_fuse_3d(in_channels=2*self.embedding_dim)
        self.cross_fuse_c2 = cross_fuse_3d(in_channels=2*self.embedding_dim)
        self.cross_fuse_c1 = cross_fuse_3d(in_channels=2*self.embedding_dim)


        #taking outputs from middle of the encoder
        self.sce = semantic_context_enhance()
        self.in_channels = [64, 128, 320, 512]

        self.make_pred_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=2,sigmoid = False)
        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        self.linear_fuse_bcd = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.dense_2x_bcd   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.dense_1x_bcd   = nn.Sequential( ResidualBlock(self.embedding_dim))  
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Softmax() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape
        outputs = []
        
        
        
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        diff_feat_c4 = self.cross_fuse_c4([_c4_1, _c4_2])
        diff_feat_c4_up = resize(diff_feat_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        


        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        diff_feat_c3 = self.cross_fuse_c3([_c3_1, _c3_2]) + F.interpolate(diff_feat_c4, scale_factor=2, mode="bilinear")
        # diff_feat_c3 = self.sce([resize(diff_s, size=diff_feat_c3.size()[2:], mode='bilinear', align_corners=False),diff_feat_c3])
        diff_feat_c3_up = resize(diff_feat_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
                # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.cross_fuse_c2([_c2_1, _c2_2]) + F.interpolate(diff_feat_c3, scale_factor=2, mode="bilinear")
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.cross_fuse_c1([_c1_1, _c1_2]) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

        #Linear Fusion of difference image from all scales
        _c_bcd = self.linear_fuse_bcd(torch.cat((diff_feat_c4_up, diff_feat_c3_up,_c2_up,_c1), dim=1))
        x = self.dense_2x(_c_bcd)
        x = self.dense_1x(x)
        p_bcd = self.make_pred_bcd(x)


        

        outputs.append(p_bcd)  
        return outputs
    
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet18
class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = nn.GELU()

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(ffn_drop)

    def forward(self, x, identity=None):

        out = self.layers(x)
        if identity is None:
            identity = x
        return identity
class Conv3d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[7,7,3], stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv3d_cd, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # Calculating the kernel_diff for 3D
            [C_out, C_in, kernel_size, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
class cross_feature_3d(nn.Module):
    def __init__(self, in_channels):
        super(cross_feature_3d, self).__init__()
        self.conv3d = nn.Sequential(
            Conv3d_cd(1, 1, kernel_size=[3,3,3], stride=1, padding=[1,1,1], bias=True, theta= 0.7),
            # nn.InstanceNorm3d(1),
            nn.GELU(),
            Conv3d_cd(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True, theta= 0.7),
            # nn.InstanceNorm3d(1),
            nn.GELU(),

        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.InstanceNorm2d(in_channels),
            nn.GELU(),
        )
        # self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b,c,h,w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w )
        tensor2 = tensor2.view(b, c, h*w )
        cross_x = torch.cat((tensor1, tensor2), dim=2)
        cross_x = cross_x.view(b, c*2, h,w)
        cross_x = cross_x.unsqueeze(1)
        cross_x = self.conv3d(cross_x)
        cross_x = cross_x.squeeze(1)   
        cross_x = self.fuse_conv(cross_x)
        # cross_x = self.dropout(cross_x)
        return cross_x    
class TemporalAttention(nn.Module):
    def __init__(self, in_channels, time_steps):
        super(TemporalAttention, self).__init__()
        # 使用1x1卷积相当于在每个时间步使用一个全连接层
        self.time_attention = nn.Conv1d(in_channels, time_steps, kernel_size=1)

    def forward(self, x):
        # x: [B, T, C, H, W]，假设T是时间维度
        # 首先调整维度为[B, C, T, H, W]以适应Conv1d
        # x = x.permute(0, 2, 1, 3, 4)
        b, c, t, h, w = x.shape
        # 调整维度为[B*C, T, H*W]以进行时间注意力操作
        x = x.reshape(b * c, t, h * w)
        # 应用时间注意力机制
        attention = self.time_attention(x)
        # 应用softmax
        attention = F.softmax(attention, dim=1)
        # 将注意力权重应用到原始输入
        x = x * attention
        # 恢复维度为[B, C, T, H, W]
        x = x.view(b, c, t, h, w)
        # 将维度调整回[B, T, C, H, W]
        # x = x.permute(0, 2, 1, 3, 4)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 使用2D卷积来学习空间位置的重要性
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.view(b * c, t, h, w)
        # 计算空间注意力权重
        attention = self.spatial_attention(x)
        # 应用sigmoid函数获取权重因子
        attention = torch.sigmoid(attention)
        # 将注意力权重应用到原始输入
        x = x * attention
        x = x.view(b, c, t, h, w)
        x = x.view(b, c*t, h, w)
        return x

class cross_fuse_3d_atten(nn.Module):
    def __init__(self, in_channels):
        super(cross_fuse_3d_atten, self).__init__()
        self.conv3d1 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
                                nn.GELU())                 
        self.conv3d2 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,5,5], stride=1, padding=(1, 2, 2), bias=True),
                                nn.GELU())      
        self.conv3d3 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,7,7], stride=1, padding=(1, 3, 3), bias=True),
                                nn.GELU())     
        self.temporal_attention = TemporalAttention(in_channels,in_channels)
        self.spatial_attention = SpatialAttention(in_channels)
        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(in_channels//2),
        # )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
        )
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
        )
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        tensor1 = x[0]
        tensor2 = x[1]
        b,c,h,w = tensor1.shape
        tensor1 = tensor1.view(b, c, h*w )
        tensor2 = tensor2.view(b, c, h*w )
        cross_x = torch.cat((tensor1, tensor2), dim=2)
        cross_x = cross_x.view(b, c*2, h,w)
        cross_x = cross_x.unsqueeze(1)
        cross_x1 = self.conv3d1(cross_x)
        # cross_x1 = cross_x1.squeeze(1)
        cross_x2 = self.conv3d2(cross_x)
        # cross_x2 = cross_x2.squeeze(1)
        cross_x3 = self.conv3d3(cross_x)
        cross_x = torch.cat((cross_x1, cross_x2, cross_x3), dim=1)
        cross_x = self.temporal_attention(cross_x)
        cross_x = self.spatial_attention(cross_x)
        # cross_x = cross_x.squeeze(1)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)
        cross_x = self.fuse_conv1(cross_x)
        cross_x = self.dropout(cross_x)
        return cross_x
    
# class cdsc(nn.Module):

#     def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
#         super(cdsc, self).__init__()
#         self.channels = 64
#         #ResNet18 Backbone
#         self.backbone1 = resnet18(pretrained=True)
#         self.backbone2 = resnet18(pretrained=True)
#         # Remove the final classification layer
#         self.backbone1.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.backbone2.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.diff = cross_feature_3d(
#                     in_channels=self.channels,)
#         self.backbone1 = nn.Sequential(*list(self.backbone1.children())[:-2])
#         self.backbone2 = nn.Sequential(*list(self.backbone2.children())[:-2])
#         self.in_channels = [64, 128, 256, 512]
#         self.dropout = nn.Dropout(0.1)
#         self.convs = nn.ModuleList()
#         for i in range(4):
#             self.convs.append(nn.Sequential(nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.InstanceNorm2d(self.channels),
#             nn.GELU()))     
#         # Freeze the backbone layers
#         for param in self.backbone1.parameters():
#             param.requires_grad = False
#         for param in self.backbone2.parameters():
#             param.requires_grad = False
#         # Custom Decoder
#         self.fusion_conv = nn.Sequential(
#                 nn.Conv2d(self.channels*4, self.channels//2, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.InstanceNorm2d(self.channels//2),
#                 nn.GELU())
#         self.discriminator = MixFFN(
#             embed_dims=self.channels,
#             feedforward_channels=self.channels,
#             ffn_drop=0.,
#             dropout_layer=dict(type='DropPath', drop_prob=0.2),
#             act_cfg=dict(type='GELU'))
#         self.conv_seg = nn.Conv2d(self.channels, 2, kernel_size=1)
#     def base_forward(self, inputs):
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             # print(idx,x.shape)
#             # print(idx,conv(x).shape)
#             # print(idx,conv(x).shape)
#             if idx==0:
#                 detail_feature = conv(x)
#             outs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode='bilinear'))
        
#         out = self.fusion_conv(torch.cat(outs, dim=1))
#         return out
#     def cls_seg(self, feat):
#         """Classify each pixel."""
#         if self.dropout is not None:
#             feat = self.dropout(feat)
#         output = self.conv_seg(feat)
#         return output    
#     def forward(self, x1, x2):
#         x_size = x1.size()
#         # Forward pass through ResNet backbones
#         fx1,fx2 = [],[]
#         for layer in self.backbone1:
#             x1 = layer(x1)
#             fx1.append(x1)
#         fx1 = fx1[-4:]
#         fx1 = self.base_forward(fx1)
#         for layer in self.backbone2:
#             x2 = layer(x2)
#             fx2.append(x2)
#         fx2 = fx2[-4:]
#         fx2 = self.base_forward(fx2)
#         # print(fx1.shape,fx2.shape)
        
#         out = self.diff([fx1, fx2])
#         out = self.discriminator(out)
#         out = self.cls_seg(out)
#         # print(out.shape)
#         return F.upsample(out, x_size[2:], mode='bilinear'), [fx1, fx2]




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 16, 128, 128]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 256, 8, 8]
            nn.ReLU(),
            nn.Flatten()  # 输出: [batch_size, 256*8*8]
        )
        self.fc_mu = nn.Linear(256*8*8, 100)
        self.fc_logvar = nn.Linear(256*8*8, 100)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(100, 256*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch_size, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch_size, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch_size, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch_size, 16, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: [batch_size, 1, 256, 256]
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class ShallowResNet(nn.Module):
    def __init__(self):
        super(ShallowResNet, self).__init__()
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Create a residual block with downsample
        self.res_block = ResidualBlock1(64, 64, stride=2, 
                                       downsample=nn.Sequential(
                                           nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
                                           nn.BatchNorm2d(64)
                                       ))

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.res_block(x)
        return x

class hetecd(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(hetecd, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 4, 6, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.2
        self.attn_drop = 0.2
        self.drop_path_rate = 0.2 
        self.backbone1 = Backbone( patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])
        self.backbone2 = Backbone( patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])
        self.vae1 = ShallowResNet()
        self.vae2 = ShallowResNet()
        # self.vae1.apply(self.init_weights)
        # self.vae2.apply(self.init_weights)
        self.CD_Decoder   = CD_3D_Decoder(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16])
        
    # def init_weights(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         nn.init.normal_(m.weight, mean=0.0, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    def forward(self, x1, x2):
        x_size = x1.size() 
        # print(x_size)
        # x1 = self.vae1(x1)
        # x2= self.vae1(x2)
        # print(x1.size())
        [fx1, fx2] = [self.backbone1(x1), self.backbone2(x2)]
        c1_1, c2_1, c3_1, c4_1 = fx1
        c1_2, c2_2, c3_2, c4_2 = fx2
        cp = self.CD_Decoder(fx1, fx2)

        return F.upsample(cp[-1], x_size[2:], mode='bilinear'),[[c3_1,c4_1],[c3_2, c4_2]]


