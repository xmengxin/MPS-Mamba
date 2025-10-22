## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from typing import Optional, Callable
from functools import partial
from einops import rearrange, repeat
import math
import torch.utils.checkpoint as checkpoint


##########################################################################
## Layer Norm



##########################################################################
##---------- Restormer -----------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
       
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W).contiguous()
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]



class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # import pdb
        # pdb.set_trace()
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):    # Residue State Space Block
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        # self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        # self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        # self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

        mlp_hidden_dim = int(hidden_dim * 2)
        self.mlp = Mlp(hidden_dim, mlp_hidden_dim, hidden_dim )



    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        # x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        # x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = input + self.drop_path(self.self_attention(x))
        x = x  + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.view(B, -1, C).contiguous()
        return x

class BasicLayer(nn.Module):# Residue State Space Group (RSSG)
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                is_light_sr=is_light_sr))
        
        # add pos encoder 
       
        self.pos_blocks = PosCNN(dim, dim)



    def forward(self, x, x_size):
        
        for i, blk in enumerate(self.blocks):
            # import pdb
            # pdb.set_trace()
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
                if i==0:
                    x = self.pos_blocks(x, x_size[0], x_size[1])
                
        
        return x

   



class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size, scal=1):
        # x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        x = x.transpose(1, 2).view(x.shape[0], -1, int(x_size[0]/scal), int(x_size[1]/scal))  # b , c, Ph, Pw
        return x

    def flops(self):
        flops = 0
        return flops

#**************************Downsample *******************************#
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)
#**************************END**********************************#
    


#**************************upsample *******************************#
class Upsample_2(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
#**************************END**********************************#


class MambaRestormer(nn.Module):
    
    def __init__(self,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 drop_rate=0.,

                 img_size=64,
                 patch_size=1,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                
                 inp_channels=3, 
                 out_channels=3, 
                 dim=48,
                 num_blocks = [4,6,6,8], 
                 num_refinement_blocks = 4,
                 ffn_expansion_factor = 2.66,
                 bias = False,
                 dual_pixel_task = False, ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 LayerNorm_type = 'WithBias',   ## Other option 'BiasFree ,       
                    **kwargs):
        super(MambaRestormer, self).__init__()

        num_in_ch = inp_channels
        # num_out_ch = inp_channels
        # num_feat = 64
        
        self.img_range = img_range
        if inp_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, dim, 3, 1, 1)

        # # ------------------------- 2, deep feature extraction ------------------------- #
        
        self.embed_dim = dim
        self.patch_norm = patch_norm
        self.num_features = dim

        self.pos_drop = nn.Dropout(p=drop_rate)


        # # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
      
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans= dim,
            embed_dim= dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = BasicLayer(
            dim=dim,
            depth=num_blocks[0],
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)
      
        
         #*********多尺度图像卷积*************#
        self.img_down_sample_2 = nn.Conv2d(inp_channels, 2*dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.input_reduce_chan_level2 = nn.Conv2d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)

        self.encoder_level2 = BasicLayer(
            dim=dim*2,
            depth=num_blocks[1],
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)
     
        
        self.img_down_sample_3 = nn.Conv2d(inp_channels, 3*dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.input_reduce_chan_level3 = nn.Conv2d(int(dim*7), int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = BasicLayer(
            dim=dim*4,
            depth=num_blocks[2],
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # self.img_down_sample_4 = nn.Conv2d(inp_channels, 4*dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.input_reduce_chan_level4 = nn.Conv2d(int(dim*12), int(dim*2**3), kernel_size=1, bias=bias)
        # self.latent = BasicLayer(
        #     dim=dim*8,
        #     depth=num_blocks[3],
        #     d_state = d_state,
        #     mlp_ratio=mlp_ratio,
        #     drop_path=drop_path,
        #     norm_layer=norm_layer,
        #     downsample=downsample,
        #     use_checkpoint=use_checkpoint,
        #     is_light_sr = is_light_sr)
        
        # self.img_upsample_4 = nn.Conv2d(dim*2**3, 3, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.up4_3 = Upsample_2(int(dim*2**3)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = BasicLayer(
        #     dim=dim*4,
        #     depth=num_blocks[2],
        #     d_state = d_state,
        #     mlp_ratio=mlp_ratio,
        #     drop_path=drop_path,
        #     norm_layer=norm_layer,
        #     downsample=downsample,
        #     use_checkpoint=use_checkpoint,
        #     is_light_sr = is_light_sr)

        self.img_upsample_3 = nn.Conv2d(dim*2**2, 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up3_2 = Upsample_2(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer(
            dim=dim*2,
            depth=num_blocks[1],
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)
        
        self.img_upsample_2 = nn.Conv2d(dim*2**1, 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up2_1 = Upsample_2(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer(
            dim=dim*2,
            depth=num_blocks[0],
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)
        
        # self.refinement = BasicLayer(
        #     dim=dim*2,
        #     depth=num_refinement_blocks,
        #     d_state = d_state,
        #     mlp_ratio=mlp_ratio,
        #     drop_path=drop_path,
        #     norm_layer=norm_layer,
        #     downsample=downsample,
        #     use_checkpoint=use_checkpoint,
        #     is_light_sr = is_light_sr)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    #********after vssblocks patch and unpatch*************#
        self.patch_embed2 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed2 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
    

    def forward(self, inp_img):
       
        
        x = self.conv_first(inp_img)

        #**********flatten the image as the squence***********
        x_size = (x.shape[2], x.shape[3])
        x_patch = self.patch_embed(x) # N,L,C
        x_patch = self.pos_drop( x_patch)
        #*****************************************************

        #************* encoder  first level ******************
        inp_enc_level1 = x_patch  #self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, x_size)
        out_enc_level1 = self.patch_unembed2(out_enc_level1, x_size) # patch to image
        #*****************************************************
        
        #*************encoder  second level ******************
        # import pdb 
        # pdb.set_trace()
        down_img_level2 = F.interpolate(inp_img, scale_factor=0.5, mode='bilinear', align_corners=True) #** 下采样图像
        down_img_level2_conv =  self.img_down_sample_2(down_img_level2)
       
        inp_enc_level2 = self.down1_2(out_enc_level1) #** 下采样 last level
        inp_enc_level2 = torch.cat([inp_enc_level2, down_img_level2_conv], 1) # 拼接多尺度
        inp_enc_level2 = self.input_reduce_chan_level2(inp_enc_level2) # reduce channel 为 2*c

        inp_enc_level2 = self.patch_embed2(inp_enc_level2) #*****图像到patch序列
        out_enc_level2 = self.encoder_level2(inp_enc_level2, (int(x_size[0]/2), int(x_size[1]/2))) # vssb blocks 
        out_enc_level2 = self.patch_unembed2(out_enc_level2, x_size, 2)  #序列到图像
        #*****************************************************

        #*************encoder  third  level ******************
        down_img_level3 = F.interpolate(inp_img, scale_factor=0.25, mode='bilinear', align_corners=True) #** 下采样图像
        down_img_level3_conv  =  self.img_down_sample_3(down_img_level3)
       
        inp_enc_level3 = self.down2_3(out_enc_level2)#** 下采样 last level
        inp_enc_level3 = torch.cat([inp_enc_level3, down_img_level3_conv], 1) # 拼接多尺度
        inp_enc_level3 = self.input_reduce_chan_level3(inp_enc_level3) # reduce channel 为 4*c

        inp_enc_level3 = self.patch_embed2(inp_enc_level3) #*****图像到patch序列
        out_enc_level3 = self.encoder_level3(inp_enc_level3, (int(x_size[0]/4), int(x_size[1]/4))) 
        out_enc_level3 = self.patch_unembed2(out_enc_level3, x_size, 4)  #序列到图像

        #*****************************************************

        #*************encoder  fourth level ******************
        # down_img_level4 = F.interpolate(inp_img, scale_factor=0.125, mode='bilinear', align_corners=True) #** 下采样图像
        # down_img_level4_conv =  self.img_down_sample_4(down_img_level4)
        #
        # inp_enc_level4 = self.down3_4(out_enc_level3)   #** 下采样 last level
        # inp_enc_level4 = torch.cat([inp_enc_level4,  down_img_level4_conv], 1) # 拼接多尺度
        # inp_enc_level4 = self.input_reduce_chan_level4(inp_enc_level4) # reduce channel 为 8*c

        
        # inp_enc_level4 = self.patch_embed2(inp_enc_level4) #*****图像到patch序列
        # latent = self.latent(inp_enc_level4, (int(x_size[0]/8), int(x_size[1]/8))) # vssb blocks
        # latent = self.patch_unembed2(latent, x_size, 8)  #序列到图像 8c

        # #****oupt image_Leve 4
        # latent_conv = self.img_upsample_4(latent)   # reduce channel 8c 为 3
        # out_img_4 = latent_conv + down_img_level4  # h/8,w/8, 3
        #*****************************************************

        #************* decoder  first level ******************
        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # inp_dec_level3 = self.patch_embed2(inp_dec_level3) #*****图像到patch序列
        # out_dec_level3 = self.decoder_level3(inp_dec_level3, (int(x_size[0]/4), int(x_size[1]/4))) # vssb blocks
        # out_dec_level3 =  self.patch_unembed2(out_dec_level3, x_size, 4)  #序列到图像 # h/4,w/4, 4c

        #****oupt image_Leve 3
        out_dec_level3_conv = self.img_upsample_3(out_enc_level3)
        out_img_3 = out_dec_level3_conv + down_img_level3  
        #*****************************************************

         #************* decoder  second  level ******************
        inp_dec_level2 = self.up3_2(out_enc_level3) #*****上采样
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self.patch_embed2(inp_dec_level2) #*****图像到patch序列
        out_dec_level2 = self.decoder_level2(inp_dec_level2, (int(x_size[0]/2), int(x_size[1]/2))) 
        out_dec_level2 =  self.patch_unembed2(out_dec_level2, x_size, 2)  #序列到图像

        # #****oupt image_Leve 3
        out_dec_level2_conv = self.img_upsample_2(out_dec_level2)
        out_img_2 = out_dec_level2_conv + down_img_level2
        #*****************************************************

        #************* decoder  3  level ******************
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.patch_embed2(inp_dec_level1) #*****图像到patch序列
        out_dec_level1 = self.decoder_level1(inp_dec_level1, x_size)
        # out_dec_level1 =  self.patch_unembed2(out_dec_level1)  #序列到图像
        
        # out_dec_level1 = self.refinement(out_dec_level1, x_size) # 输出序列

        #### For Dual-Pixel Defocus Deblurring Task ####
        out_dec_level1 = self.patch_unembed(out_dec_level1, x_size) #序列到图像
       
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(x)
            out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        else:
           
            out_dec_level1 = self.output(out_dec_level1) + inp_img
      
        

        return (out_img_3, out_img_2, out_dec_level1)

        # return out_dec_level1