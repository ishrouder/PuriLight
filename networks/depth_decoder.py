from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.layers import trunc_normal_



class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, scale="n", freeze=0,
                 opt=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.upsample_mode = "bilinear"  # "bilinear"#'nearest'
        if opt is not None:
            if opt.upsample == "nearest":
                self.upsample_mode = "nearest"
            elif opt.upsample == "bilinear":
                self.upsample_mode = "bilinear"
            elif opt.upsample == "dysample_pl":
                self.upsample_mode = "dysample_pl"
            elif opt.upsample == "dysample_lp":
                self.upsample_mode = "dysample_lp"

        if self.upsample_mode in ["dysample_pl", "dysample_lp"]:
            self.use_all_upsample_learn = True
            self.use_dysample = True
        else:
            self.use_all_upsample_learn = False

        self.scales = scales
        group = 4


        self.num_ch_enc = num_ch_enc
        self.freeze = freeze

        self.convs = OrderedDict()
        num_ch_in = self.num_ch_enc[-1]

        self.convs[("upconv_upsample", 3, 0)] = DySample(num_ch_in, style="pl",groups=group)
        num_ch_in += self.num_ch_enc[1]           
        num_ch_out = self.num_ch_enc[1]           

        self.convs[("upconv", 3, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)

        self.convs[("upconv_upsample", 2, 0)] = DySample(num_ch_out, style="pl",groups=2)

        num_ch_in = num_ch_out + self.num_ch_enc[0]     
        num_ch_out = (num_ch_in / 2).astype('int')      


        self.convs[("upconv", 2, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)

        self.convs[("upconv_upsample", 1, 0)] = DySample(num_ch_out, style="pl",groups=4)

        num_ch_in = num_ch_out          
        num_ch_out = self.num_ch_enc[0]  


        self.convs[("upconv", 1, 1)] = C3k2(num_ch_in, num_ch_out,c3k=True)



        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_enc[0],
                                              self.num_output_channels)

        self.convs[(f"upsample_with_params_0", 1)] = DySample(self.num_ch_enc[0], style="pl", groups=group)



        self.convs[("dispconv", 2)] = Conv3x3(num_ch_in,
                                              self.num_output_channels)
        self.convs[(f"upsample_with_params_0", 2)] = DySample(num_ch_in, style="pl", groups=1)


        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_enc[1],
                                              self.num_output_channels)
        self.convs[(f"upsample_with_params_0", 3)] = DySample(self.num_ch_enc[1], style="pl", groups=1)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.outputs = {}

        self.scale_factor = 2.0

    def upsample(self, x: torch.Tensor, mode="nearest"):

        return F.interpolate(x, scale_factor=self.scale_factor, mode=mode)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1] 

        x = upsample(x, mode=self.upsample_mode)
        x = [self.convs[("upconv_upsample", 3, 0)](x)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 3)](x)
        f = self.convs[("dispconv", 3)](f)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 2)] = self.sigmoid(f)

        x = upsample(x, mode=self.upsample_mode)
        x = [self.convs[("upconv_upsample", 2, 0)](x)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 2)](x)
        f = self.convs[("dispconv", 2)](f)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 1)] = self.sigmoid(f)

        x = upsample(x, mode=self.upsample_mode)
        x = self.convs[("upconv_upsample", 1, 0)](x)
        # x += [input_features[0]]
        # x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        f = self.convs[(f"upsample_with_params_{0}", 1)](x)
        f = self.convs[("dispconv", 1)](f)
        f = self.upsample(f, mode='bilinear')

        outputs[("disp", 0)] = self.sigmoid(f)

        return outputs

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
class DySample(nn.Module):
    def __init__(self, in_channels, scale=1, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

def upsample(x,scale_factor=2,mode="nearest"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)