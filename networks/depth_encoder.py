import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output
    
    
class DropPath(nn.Module):

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'
    
    
def drop_path(x, drop_prob = 0., training=False, scale_by_keep=True):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.data_format == "channels_last":
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return x.permute(0, 3, 1, 2).contiguous()
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            
                   
class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x
  
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class SDC(nn.Module):
    def __init__(self, inp, oup, *, ksize, stride, drop_path):
        super(SDC, self).__init__()
        self.stride = stride
        assert stride in [1, 2]


        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        dim = inp // 2

        mid_channels = outputs = int(oup //2) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.branch_main_1 = nn.Sequential(
            # pw
            nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            LayerNorm(mid_channels, eps=1e-6),)
            
        self.branch_main_2 = nn.Sequential(  
            
            nn.Linear(mid_channels, 6 * mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6 * mid_channels, mid_channels),)
        
        self.branch_main_3 = nn.Sequential(   
            nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            LayerNorm(mid_channels, eps=1e-6),)
            
        self.branch_main_4 = nn.Sequential(
            nn.Linear(mid_channels, 6 * mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6 * mid_channels, mid_channels),)
        
        self.branch_main_5 = nn.Sequential(    
            nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            LayerNorm(mid_channels, eps=1e-6),)
            
        self.branch_main_6 = nn.Sequential(
            nn.Linear(mid_channels, 6 * mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6 * mid_channels, mid_channels),)
            
            
    @staticmethod
    def channel_shu(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        
        return x


    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)

        input = x  
        x = x.contiguous() 
        
        x = self.branch_main_1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.branch_main_2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        
        x = self.branch_main_3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.branch_main_4(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.branch_main_5(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.branch_main_6(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = input + self.drop_path(x)
        x = torch.cat((x, x_proj), 1)
        out = self.channel_shu(x, 2)   
        return out


    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
      

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, num):
        super(AttentionGate, self).__init__()
        kernel_size = num 
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self,num, dim, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate(num)
        self.hc = AttentionGate(num)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate(num)
            
        self.conv1 = conv3x3(dim, dim, stride = 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        
        
    def forward(self, x):

        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class ZPool11(nn.Module):
    def forward(self, x):
        real = torch.cat((torch.max(x.real, 1)[0].unsqueeze(1), torch.mean(x.real, 1).unsqueeze(1)), dim=1)
        imag = torch.cat((torch.max(x.imag, 1)[0].unsqueeze(1), torch.mean(x.imag, 1).unsqueeze(1)), dim=1)
        return torch.cat((real, imag), dim=1)


class RAKA(nn.Module):
    
    def __init__(self):
        super(RAKA, self).__init__()
        kernel_size = 7
        self.compress = ZPool11()
        self.conv = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False, bn=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class FDMP(nn.Module):
    def __init__(self, channel, drop_path=0.3, cutoff_ratio=0.95):
        super().__init__()
        
        C0 = int(channel / 2)
    
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)  #3*3
        self.conv_1 = nn.Conv2d(C0, C0, kernel_size=1)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        
        self.bng_real = BNGELU(C0)
        self.bng_imag = BNGELU(C0)

        self.cw = RAKA()
        self.hc = RAKA()
        self.hw = RAKA()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cutoff_ratio = cutoff_ratio
        self.cutoff_logits = nn.Parameter(torch.tensor(0.0))
        
    @staticmethod
    def channel_shu_glo(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        
        return x

    def forward(self, x):
        bias = x
        x0, x1 = self.channel_shuffle(x)
        
        x_0 = self.conv_0(x0)

        x_fft = torch.fft.rfft2(x1, dim=(2, 3), norm="ortho")
        
        real = x_fft.real
        imag = x_fft.imag
        
        H = 2 * (real.shape[2] -1)
        W = real.shape[3]

        freq_H = torch.fft.rfftfreq(H, d=1.0)
        freq_W = torch.fft.fftfreq(W, d=1.0)
        freq_H_grid, freq_W_grid = torch.meshgrid(freq_H, freq_W, indexing="ij")
        freq_radius = torch.sqrt(freq_H_grid**2 + freq_W_grid**2)

        cutoff = self.cutoff_ratio * freq_radius.max()
                
        low_pass_mask = (freq_radius <= cutoff).to(real.device)  #

        real = real * low_pass_mask
        imag = imag * low_pass_mask

        real = self.conv_1(real)
        imag = self.conv_1(imag)
        real = self.bng_real(real)
        imag = self.bng_imag(imag)

        x = torch.complex(real, imag)

        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        x_out = self.hw(x)
        x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        
        x = torch.fft.irfft2(x_out, dim=(2, 3), norm="ortho")
        output = torch.cat([x_0, x], dim=1)
        output = self.conv_2(output)
        output = self.channel_shu_glo(output, 2)
        x = self.drop_path(output) + bias

        return x       
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
    

class PuriLightX(nn.Module):
    """
    PuriLightX
    """
    def __init__(self, in_chans=3, model='purilightx-lp', drop_path_rate=0.3, **kwargs):

        super().__init__()

        if model == 'purilightx-lp':
            self.num_ch_enc = np.array([80, 160, 188])
            self.dims = [80, 160, 188]
        elif model == 'purilightx-lf':
            self.num_ch_enc = np.array([72, 160, 208])
            self.dims = [72, 160, 208]
        else:
            raise ValueError("Unsupported model: {}".format(model))

        self.k = [0, 1, 2]


        self.downsample_layers = nn.ModuleList()  
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1) 


        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)
            
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]

        self.stages = nn.ModuleList()
        for i in range(3):
            stage_blocks=[]
            stage_blocks.append(SDC(self.dims[i], self.dims[i], ksize=3, stride=1, drop_path=dp_rates[i]))
            stage_blocks.append(LayerNorm(self.dims[i])) 
            if i == 2:
                stage_blocks.append(FDMP(self.dims[i]))
            self.stages.append(nn.Sequential(*stage_blocks))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225
    
        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
            
        x = self.stages[0][-1](x)

        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)
         

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)

            x = self.stages[i][-1](x)

            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x