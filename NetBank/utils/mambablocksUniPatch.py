import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

try:
    from .csms6s import selective_scan_step,CrossScan, CrossMerge,CrossScan_basic, CrossMerge_basic,CrossMerge_plus_uni,CrossScan_plus_uni
    from .csms6s import SelectiveScanCore
except:

    from csms6s import selective_scan_step,CrossScan, CrossMerge
    from csms6s import SelectiveScanCore

# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D



class BiSTSSM_v2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=5, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        joints=5,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.d_inner=d_inner
        self.joints=joints
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)


        assert out_norm_dwconv3 is False and out_norm_none is False and out_norm_sigmoid is False and out_norm_softmax is False
        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)


        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_basic, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_basic),
            v2_uniDirection=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_plus_uni, SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_plus_uni),
            )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 2
        if forward_type is 'v2_uniDirection':
            k_group = 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        # conv =======================================
        self.d_conv = d_conv
        self.pad_t = (d_conv - 1)          # time causal left pad
        self.pad_j = (d_conv - 1) // 2   
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=0,#((d_conv - 1),(d_conv - 1) // 2),
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner*joints, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        
        self.gatingConv = nn.Conv1d(d_inner*joints*2, d_inner*joints, kernel_size=1)
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # self.cov1d 
        self.d_state = d_state
        self.dt_rank = dt_rank
        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner*joints, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner*joints, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner*joints, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanCore,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=False,
        **kwargs,
    ):
        '''
        partial(self.forward_corev2, 
                force_fp32=(not self.disable_force32), 
                CrossScan=CrossScan_plus_poselimbs, 
                SelectiveScan=SelectiveScanCore, 
                CrossMerge=CrossMerge_plus_poselimbs)
        '''

        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        x = x.transpose(-1,-2)
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)
        
        if cascade2d:
            raise
            
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                raise
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                # xs.shape (B, 2, D, L)
                # x_proj_weight.shape (4, , D)
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float) # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)
        y = y.transpose(1,2)
        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        if not self.disable_z:
            # print(x.shape)
            # chunk x into two parts, z and x
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            #window size 5
            x = F.pad(x, (self.pad_j, self.pad_j, self.pad_t, 0))
            x = self.conv2d(x)  # output time length == original T

        x = self.act(x)
        # torch.Size([1, 256, 243, 17])
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out


    def forward_corev2_step_uni(
        self,
        x: torch.Tensor,
        force_fp32: bool = False,
        delta_softplus: bool = True,
    ):
        """
        Streaming step for forward_type='v2_uniDirection' (K=1).
        Input x: (B, C, 1, J)  where C = d_inner, J = joints
        Output: (B, J, C)      single-frame output aligned with forward_corev2
        """
        assert x.dim() == 4, f"expect (B,C,1,J), got {x.shape}"
        B, C, T1, J = x.shape
        assert T1 == 1, "step expects a single time step (T=1)"
        assert C == self.d_inner, f"C must be d_inner={self.d_inner}, got {C}"
        assert J == self.joints, f"J must be joints={self.joints}, got {J}"

        # ---------- match forward_corev2 pre-processing ----------
        # forward_corev2 does: x = x.transpose(-1, -2)  (swap H/W)
        # Here x is (B,C,1,J) -> (B,C,J,1)
        x = x.transpose(-1, -2).contiguous()  # (B, C, J, 1)
        _, _, H, W = x.shape                  # H=J, W=1
        L = W                                 # L=1

        # K = 1 for uniDirection
        # CrossScan_plus_uni.forward flattens (C,H) into D_total and keeps W
        xs = CrossScan_plus_uni.apply(x)      # (B, 1, C*H, W) = (B,1,D_total,1)
        # D_total == d_inner * joints
        D_total = xs.shape[2]
        assert D_total == self.d_inner * self.joints

        # ---------- projections (same math as forward_corev2, but L=1, K=1) ----------
        x_proj_weight = self.x_proj_weight    # (K=1, Cproj, D_total)
        dt_projs_weight = self.dt_projs_weight  # (K=1, D_total, R)
        dt_projs_bias = self.dt_projs_bias      # (K=1, D_total)
        A_logs = self.A_logs                  # (K*D_total, d_state) -> (D_total, d_state)
        Ds = self.Ds                          # (K*D_total,) -> (D_total,)

        R = self.dt_rank
        N = self.d_state

        # xs: (B,1,D_total,1), weight: (1, R+2N, D_total)
        # => x_dbl: (B,1, R+2N, 1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

        # split along c-dim
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)   # each (B,1,*,1)

        # dts: (B,1,R,1), dt_projs_weight: (1, D_total, R)
        # => (B,1,D_total,1)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        # ---------- squeeze L=1 to per-step vectors ----------
        # u_t, delta_t: (B, D_total)
        u_t = xs[:, 0, :, 0].contiguous()
        delta_t = dts[:, 0, :, 0].contiguous()

        # B_t, C_t: (B, K=1, N=d_state)
        B_t = Bs[:, 0, :, 0].contiguous().unsqueeze(1)  # (B,1,N)
        C_t = Cs[:, 0, :, 0].contiguous().unsqueeze(1)  # (B,1,N)

        # params to fp32 if needed (match forward_corev2 style)
        As = -torch.exp(A_logs.to(torch.float32))         # (D_total, d_state)
        Ds_f = Ds.to(torch.float32)                       # (D_total,)
        delta_bias = dt_projs_bias.view(-1).to(torch.float32)  # (D_total,)

        if force_fp32:
            u_t = u_t.to(torch.float32)
            delta_t = delta_t.to(torch.float32)
            B_t = B_t.to(torch.float32)
            C_t = C_t.to(torch.float32)

        # ---------- init / fetch state ----------
        state = getattr(self, "_ssm_state", None)
        if (state is None) or (state.shape[0] != B) or (state.shape[1] != D_total) or (state.shape[2] != N) or (state.device != u_t.device):
            # keep state in fp32 for numerical stability
            state = torch.zeros((B, D_total, N), device=u_t.device, dtype=torch.float32)
            self._ssm_state = state

        # ---------- selective scan step ----------
        # returns y_t: (B, D_total) typically fp32
        y_t = selective_scan_step(
            u_t, delta_t, As, B_t, C_t, state,
            D=Ds_f, delta_bias=delta_bias,
            delta_softplus=delta_softplus
        )

        # ---------- reshape back (match forward_corev2 post-processing) ----------
        # In forward_corev2:
        # ys -> CrossMerge -> y.view(B,-1,H,W) -> if not channel_first -> (B,H,W,C) -> out_norm -> transpose(1,2)
        y_t = y_t.to(x.dtype)  # cast back to model dtype (fp16/bf16 ok)

        # (B, D_total) -> (B, d_inner, H, W=1)
        y = y_t.view(B, self.d_inner, H, 1)  # (B, C, J, 1)

        # channel_last path (your model uses channel_first=False)
        # (B, C, J, 1) -> (B, J, 1, C)
        y = y.permute(0, 2, 3, 1).contiguous()

        # out_norm is LayerNorm(d_inner) when channel_first=False
        y = self.out_norm(y)   # (B, J, 1, C)

        # forward_corev2 ends with y = y.transpose(1,2)  -> (B, 1, J, C)
        y = y.transpose(1, 2).contiguous()  # (B, 1, J, C)

        # step returns single frame: (B, J, C)
        return y[:, 0]

    def step(self, x: torch.Tensor, **kwards):
        """
        x: (B, J, d_model)  —— 单帧输入
        return: (B, J, d_model)
        """
        B, J, _ = x.shape

        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)

        # ---- depthwise conv over (time, joint) ----
        # convert to (B, C, 1, J)
        if not self.channel_first:
            # x is (B, J, C)
            x_cf = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, C, 1, J)
        else:
            # if channel_first you need to define your single-frame format; most of your code is channel_last
            raise NotImplementedError("step only implemented for channel_first=False")

        if self.with_dconv:
            buf = getattr(self, "_dconv_buf", None)
            if (buf is None) :
                # buffer holds last d_conv frames: (B, C, d_conv, J)
                buf = torch.zeros((1, self.d_inner, self.d_conv, self.joints), device=x_cf.device, dtype=torch.float32)
                self._dconv_buf = buf
                self._dconv_inited = False

            # init: keep zeros on history (causal zero state)
            # if you want "replicate first frame" init, change below init logic.
            if not getattr(self, "_dconv_inited", False):
                # shift in current frame once
                buf[:, :, :-1, :].zero_()
                buf[:, :, -1, :].copy_(x_cf[:, :, 0, :])
                self._dconv_inited = True
            else:
                buf[:, :, 0:-1, :].copy_(buf[:, :, 1:, :].clone())
                buf[:, :, -1, :].copy_(x_cf[:, :, 0, :])

            # pad and conv
            u = F.pad(buf, (self.pad_j, self.pad_j, self.pad_t, 0))   # (B,C, d_conv+pad_t, J+2*pad_j)
            u = self.conv2d(u)                                       # (B,C, d_conv, J)
            # take the output corresponding to "current time" => last time index
            x_cf = u[:, :, -1:, :]  # (B, C, 1, J)

        x_cf = self.act(x_cf)

        # ---- selective scan step (TODO: connect your selective_scan_step.cu) ----
        # x_cf currently is (B, C, 1, J). forward_core wants a 4D tensor.
        # In your forward_corev2, time is W after transpose(-1,-2). Right now time length is 1.
        #
        # You need a function that updates internal SSM state for one new timestep and returns y_t.
        #
        # Suggested API you implement in csms6s:
        #   y = SelectiveScanCore.step(u_t, state, params...)  -> returns y_t and updates state in-place
        #
        # For now, I keep a placeholder that simply calls forward_core on length=1 (correct but slow).
        # Replace this block with your true streaming selective scan step.
        x_full = x_cf  # (B,C,1,J)
        y = self.forward_corev2_step_uni(x_full, force_fp32=(not self.disable_force32))# <-- SLOW fallback; replace with real step kernel
        # y should become (B, J, C) after forward_corev2 returns

        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def resetBuf(self):
        # depthwise conv buffer + selective scan state
        self._dconv_buf = None
        self._dconv_inited = False
        # 下面这些 state 你接 selective_scan_step 时会用到
        self._ssm_state = None
        self._ssm_inited = False


class BiSTSSM(nn.Module, mamba_init, BiSTSSM_v2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=5, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

class BiSTSSMBlockUniPatch(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 5,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        output_layerWise: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.output_layerWise = output_layerWise

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
            self.op2 = BiSTSSM(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
            self.conv = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(4, 1),
                # padding=(4 // 2, 0)  # 时间方向保持长度, J方向不pad
            )
            self.convAct=nn.SiLU()
            # self.fusser  = nn.Sequential(
            #     nn.Linear(2 * hidden_dim, hidden_dim),
            #     nn.Sigmoid()
            # )
            self.fuser = self.fusser = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.norm_conv=norm_layer(hidden_dim)
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)




    def patchOP(self, input):
        B,T,J,D=input.shape
        x = rearrange(input, 'b t j d -> b d t j')
        pad_len=4-1
        pad_frames = x[:, :, 0:1, :].expand(-1, -1, pad_len, -1)
        x = torch.cat([pad_frames, x], dim=2)  # 在时间维 (dim=2) 拼接
        x = self.conv(x)
        x = self.convAct(x)
        x = rearrange(x, 'b d t j -> b t j d')
        x = self.norm_conv(x)  # LayerNorm(D)
        x = self.op2(x)
        return x
    
    def _forward(self, input: torch.Tensor):
        x = input
        x1= self.op(self.norm(x))
        x2= self.patchOP(self.norm(x))
        # gate = self.fusser(torch.cat([x1, x2], dim=-1))
        # x = gate * x1 + (1 - gate) * x2
        x = self.fusser(torch.cat([x1, x2], dim=-1))
        x = input + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def patchOPStep(self, input):
        # input: (B, J, D)
        B, J, D = input.shape
        x = rearrange(input, 'b j d -> b d j')  # (B, D, J)

        buf = getattr(self, "_patch_buf", None)
        inited = getattr(self, "_patch_buf_inited", False)

        if (buf is None):
            buf = torch.empty((B, D, 4, J), device=x.device, dtype=x.dtype)
            self._patch_buf = buf
            self._patch_buf_inited = False
            inited = False

        if not inited:
            # forward 等价：左边 pad 3 个“第一帧”，并且当前也在 buffer 里
            buf.copy_(x[:, :, None, :].expand(-1, -1, 4, -1))
            self._patch_buf_inited = True
        else:
            buf[:, :, 0:3, :].copy_(buf[:, :, 1:4, :].clone())
            buf[:, :, 3, :].copy_(x)

        y = self.conv(buf)[:, :, -1, :]   # (B, D, J) 取当前时刻输出
        y = self.convAct(y)
        y = rearrange(y, 'b d j -> b j d')
        y = self.norm_conv(y)
        y = self.op2.step(y)
        return y

    def step(self, input: torch.Tensor):
        # input: (B, J, D)
        x = input
        x1 = self.op.step(self.norm(x))
        x2 = self.patchOPStep(self.norm(x))
        x = self.fusser(torch.cat([x1, x2], dim=-1))
        x = input + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def resetBuf(self):
        self._patch_buf=None
        self._patch_buf_inited=None
        self.op.resetBuf()
        self.op2.resetBuf()

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)