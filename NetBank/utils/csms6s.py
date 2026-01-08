import torch

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    print(e, flush=True)

# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)
    
class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs
    
class CrossScan_basic(torch.autograd.Function):
    @staticmethod

    def forward(ctx, x: torch.Tensor):
        # 10， 256， 200， 5
        # B: batch size
        # C: channel size
        # H: number of frames or joints
        # W: number of joints or frames
        B, C, H, W = x.shape
        # assert W == 5, 'Expected 5 joints for this version.'
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C*H, W))
        
        # 第1通道：将每个关节的值与其他关节的值相加：目前不考虑，只是展平
        xs[:, 0] = x.flatten(1, 2)

        # 第2通道：翻转前者
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs

    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # (B, C*W, H)
        y = ys.view(B, C, H, W) # Change to (B, C, H, W)
        return y
    
class CrossMerge_basic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (D, H)
        ys = ys.view(B, K, -1, W)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).view(B, -1, W)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        D, H = ctx.shape
        B, C, W = x.shape
        xs = x.new_empty((B, 2, C, W))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 2, D, H, W)
        return xs
    
class CrossMerge_plus_uni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (D, H)
        ys = ys.view(B, K, -1, W)
        y = ys[:, 0] #+ ys[:, 1].flip(dims=[-1]).view(B, -1, W)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        D, H = ctx.shape
        B, C, W = x.shape
        xs = x.new_empty((B, 1, C, W))
        xs[:, 0] = x
        #xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 1, D, H, W)
        return xs
    
class CrossScan_plus_uni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # 10， 256， 200， 5
        # B: batch size
        # C: channel size
        # H: number of frames or joints
        # W: number of joints or frames
        B, C, H, W = x.shape
        # assert W == 5, 'Expected 5 joints for this version.'
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C*H, W))
        
        # 第1通道：将每个关节的值与其他关节的值相加：目前不考虑，只是展平
        xs[:, 0] = x.flatten(1, 2)

        # # 第2通道：翻转前者
        # xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs

    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        ys = ys[:, 0] #+ ys[:, 1].flip(dims=[-1]) # (B, C*W, H)
        y = ys.view(B, C, H, W) # Change to (B, C, H, W)
        return y
    

class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
