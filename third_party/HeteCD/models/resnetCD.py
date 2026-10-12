import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18

class DoubleConv_down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_up(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                             stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2,
                                    padding=1)
        self.conv = DoubleConv_up(in_channels, out_channels)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        

    def forward(self, x):
        return nn.Tanh()(self.conv(x))
    



# class Encoder(nn.Module):
#     def __init__(self, n_channels, bilinear=False):
#         super(Encoder, self).__init__()
#         self.inc = DoubleConv_down(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         return [x1, x2, x3, x4, x5]


# class Decoder(nn.Module):
#     def __init__(self, n_classes):
#         super(Decoder, self).__init__()
#         self.conv = nn.Sequential(
#         nn.Conv2d(512+256, 128, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.BatchNorm2d(128),
#         nn.Conv2d(128, 32, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.BatchNorm2d(32),
#     )
#         self.outc = OutConv(32, n_classes)

#     def forward(self, inputs1, inputs2):
#         # print(x.shape,inputs[0].shape)
#                 # print("解码器输出")
#         for i in range(len(inputs1)):
#             print(x_opt[i].shape)
#         x1 = inputs1[-2]
#         x2 = F.interpolate(inputs1[-1], size=x1.size()[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv(x)
#         logits1 = self.outc(x)
#         x1 = inputs2[-2]
#         x2 = F.interpolate(inputs2[-1], size=x1.size()[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv(x)
#         logits2 = self.outc(x)
#         return logits1,logits2
class Decoder(nn.Module):
    def __init__(self, n_classes):
        super(Decoder, self).__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for in_channels in [64, 128, 256, 512]  # 假设 inputs1 和 inputs2 是来自 ResNet 的特征
        ])
        self.out_conv = nn.Conv2d(128, 64, kernel_size=1)  # 128 来自于拼接后的通道数 (32 * 4)
        self.outc = OutConv(64, n_classes)

    def forward(self, inputs1, inputs2):
        def process_features(inputs):
            processed_feats = []
            target_size = inputs[0].size()[2:]  # 以第一组特征的空间尺寸作为目标尺寸
            for i in range(4):
                x = self.conv1x1[i](inputs[i])
                # 调整每组特征图的大小
                if x.size()[2:] != target_size:
                    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                processed_feats.append(x)
            return processed_feats
        
        # 处理 inputs1 和 inputs2
        feats1 = process_features(inputs1)
        feats2 = process_features(inputs2)
        
        # 将处理后的特征进行拼接
        x1 = torch.cat(feats1, dim=1)  # 通道数为 32*4 = 128
        x1 = self.out_conv(x1)  # 将通道数降到 32
        logits1 = self.outc(x1)
        
        x2 = torch.cat(feats2, dim=1)  # 通道数为 32*4 = 128
        x2 = self.out_conv(x2)  # 将通道数降到 32
        logits2 = self.outc(x2)
        
        return logits1, logits2,x1,x2
#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )
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
                    print(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out
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

class CD_Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [64, 128, 256, 512], embedding_dim= 64, output_nc=2, 
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
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=c4_in_channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=c3_in_channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=c2_in_channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=c1_in_channels)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*c4_in_channels, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*c3_in_channels, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*c2_in_channels, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*c1_in_channels, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.convd1x    = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

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
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))
        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        # x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        # x = self.convd1x(x)
        # #Residual block
        # x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)
        
        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
    
class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (dict, optional): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (dict, optional): The Config for initialization.
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

        self.activate = self.build_activation_layer(act_cfg)

        in_channels = embed_dims
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        self.pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=feedforward_channels)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.drop = nn.Dropout(ffn_drop)
        
        self.dropout_layer = self.build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def build_activation_layer(self, act_cfg):
        act_type = act_cfg.get('type', 'GELU')
        if act_type == 'ReLU':
            return nn.ReLU()
        elif act_type == 'GELU':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

    def build_dropout(self, dropout_cfg):
        drop_prob = dropout_cfg.get('p', 0.5)
        return nn.Dropout(drop_prob)

    def forward(self, x, identity=None):
        out = self.fc1(x)
        out = self.pe_conv(out)
        out = self.activate(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
    
class cdsc(nn.Module):
    def __init__(self, n_channels, n_classes=2, bilinear=False):
        super(cdsc, self).__init__()
        self.encoder_opt = resnet18()
        self.encoder_sar = resnet18()
        self.gen_decoder = Decoder(n_classes)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        # self.cd_decoder   = CD_Decoder(input_transform='multiple_select', in_index=[1, 2, 3, 4], align_corners=False, 
        #              embedding_dim= 256, output_nc=2, 
        #             decoder_softmax = False, feature_strides=[2, 4, 8, 16])
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.in_channels = [64, 128, 256, 512]
        self.channels = 128
        self.fusion_conv1 = nn.Sequential(
                    nn.Conv2d(self.channels*len(self.in_channels), self.channels//2, kernel_size=1),
                    nn.BatchNorm2d(self.channels//2),
                )
        self.fusion_conv2 = nn.Sequential(
                    nn.Conv2d(self.channels*len(self.in_channels), self.channels//2, kernel_size=1),
                    nn.BatchNorm2d(self.channels//2),
                )
        self.fusion_gen_conv1 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels//2, kernel_size=1),
                    nn.BatchNorm2d(self.channels//2),
                )
        self.fusion_gen_conv2 = nn.Sequential(
                    nn.Conv2d(self.channels, self.channels//2, kernel_size=1),
                    nn.BatchNorm2d(self.channels//2),
                )
        self.conv_seg = nn.Conv2d(self.channels , 2, kernel_size=1)
        for i in range(len(self.in_channels)):
            self.convs1.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
        for i in range(len(self.in_channels)):
            self.convs2.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))             
       
    def base_forward1(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs1[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))
        out = self.fusion_conv1(torch.cat(outs, dim=1))
        return out 
    def base_forward2(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs1[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False))
        out = self.fusion_conv2(torch.cat(outs, dim=1))
        return out          
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output    
    def forward(self, x1, x2):
        x_opt = self.encoder_opt(x1)[1:]
        x_sar = self.encoder_sar(x2)[1:]
        # opt_gen,sar_gen,gan_x1,gan_x2 = self.gen_decoder(x_opt,x_sar)
        out1 = self.base_forward1(x_opt)
        out2 = self.base_forward2(x_sar)
        # out1 = self.fusion_gen_conv1(torch.cat([out1,gan_x1], dim=1))
        # out2 = self.fusion_gen_conv2(torch.cat([out2,gan_x2], dim=1))
        out = torch.cat([out1, out2], dim=1)
        out = self.discriminator(out)
        out = self.cls_seg(out)
        # print(opt_gen.shape)
        # print(out.shape)
        return out,out1
        # opt_gen,sar_gen = self.gen_decoder(x_opt,x_sar)


        # sar_gen = self.gen_decoder(x_sar)

        # print("解码器输出")
        # for i in range(len(x_opt)):
        #     print(x_opt[i].shape)
        # print("编码器输出")
        # for i in range(len(x_opt)):
        #     print(x_opt[i].shape)
        
            # out_opt = self.gen_decoder(x_opt)
            # x_opt = self.encoder_sar(out_opt)
        cp = self.cd_decoder(x_opt,x_sar)
        return cp,opt_gen,sar_gen
        # if mode == 'gen':
        #     out_opt,_ = self.gen_decoder(x_opt)
        #     out_sar,_  = self.gen_decoder(x_sar)
        #     return [out_opt,out_sar]

if __name__ == '__main__':
    model = DualEUNet(3,3)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    y = model(x1,x2)
    print(y[0].shape)