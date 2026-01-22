# from csms6s import SelectiveScanCore, selective_scan_step
# import torch
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

# Seq=1

# u = torch.randn(1, 1280, Seq,device='cuda',dtype=torch.float32)
# delta =torch.randn(1, 1280, Seq,device='cuda',dtype=torch.float32)
# A=torch.randn(1280, 16,device='cuda',dtype=torch.float32)
# B=torch.randn(1, 1, 16, Seq,device='cuda',dtype=torch.float32)
# C=torch.randn(1, 1, 16, Seq,device='cuda',dtype=torch.float32)
# D=torch.randn(1280,device='cuda',dtype=torch.float32)
# delta_bias=torch.randn(1280,device='cuda',dtype=torch.float32)
# delta_softplus=True

# state=torch.zeros(1,1280,16,device='cuda',dtype=torch.float32)

# batch_rst = SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus,)

# step_rsts=[]
# for i in range(Seq):
#     u_t=u[:,:,i]
#     delta_t=delta[:,:,i]
#     A_t=A
#     B_t=B[:,:,:,i]
#     C_t=C[:,:,:,i]
#     D_t=D
#     delta_bias_t=delta_bias
#     rst_t =selective_scan_step(u_t, delta_t, A, B_t, C_t, state, D, delta_bias, delta_softplus)
#     step_rsts.append(rst_t)
# step_rsts=torch.stack(step_rsts,dim=-1)

# diff = (batch_rst - step_rsts).abs()
# print(diff)
# print("max abs err:", diff.max().item())
# print("mean abs err:", diff.mean().item())
from csms6s import SelectiveScanCore, selective_scan_step
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

Bsz, Ddim, T = 1, 1280, 200
dstate, G = 16, 1
device = "cuda"

u = torch.randn(Bsz, Ddim, T, device=device, dtype=torch.float16)
delta = torch.randn(Bsz, Ddim, T, device=device, dtype=torch.float16)

A = torch.randn(Ddim, dstate, device=device, dtype=torch.float32)
B = torch.randn(Bsz, G, dstate, T, device=device, dtype=torch.float16)
C = torch.randn(Bsz, G, dstate, T, device=device, dtype=torch.float16)

D = torch.randn(Ddim, device=device, dtype=torch.float32)
delta_bias = torch.randn(Ddim, device=device, dtype=torch.float32)

delta_softplus = True

state = torch.zeros(Bsz, Ddim, dstate, device=device, dtype=torch.float32)
A = -torch.rand(Ddim, dstate, device=device, dtype=torch.float32)  #
delta = 0.1 * torch.randn(Bsz, Ddim, T, device=device, dtype=torch.float16)
u = 0.1 * torch.randn(Bsz, Ddim, T, device=device, dtype=torch.float16)
# batch forward
batch_rst = SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)

# step forward
step_rsts = []
for t in range(T):
    # u_t = u[:, :, t]          # [B,D]
    # delta_t = delta[:, :, t]  # [B,D]
    # B_t = B[:, :, :, t]       # [B,G,dstate]
    # C_t = C[:, :, :, t]       # [B,G,dstate]
    u_t = u[:, :, t].contiguous()
    delta_t = delta[:, :, t].contiguous()
    B_t = B[:, :, :, t].contiguous()
    C_t = C[:, :, :, t].contiguous()
    out_t = selective_scan_step(u_t, delta_t, A, B_t, C_t, state, D, delta_bias, delta_softplus)
    # 如果你的 step 返回 (out, state)，就改成：
    # out_t, state = selective_scan_step(...)

    step_rsts.append(out_t)

step_rsts = torch.stack(step_rsts, dim=-1)  # [B,D,T]
diff = (batch_rst - step_rsts).abs()
diff = (batch_rst - step_rsts).abs()
print("max:", diff.max().item())
print("mean:", diff.mean().item())
print("nan in diff:", torch.isnan(diff).any().item())