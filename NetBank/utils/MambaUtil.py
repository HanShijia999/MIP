
import torch
import torch.nn as nn
from functools import partial


from .mambablocksUniPatch import BiSTSSMBlockUniPatch
from einops import rearrange



class  VanilaMambaSelectAttenPatchWithLayerOutput(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        
        ##
        # input variable dimension: 3, 9, 3, 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans*3, embed_dim_ratio)
        # self.Temporal_patch_to_embedding = nn.Linear(in_chans*3, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        self.chunk=20

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlockUniPatch(
                hidden_dim = embed_dim, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                # forward_type='v2_plus_poselimbs'
                forward_type='v2_uniDirection',
                output_layerWise=True,
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        self.select_head = nn.Sequential(
            nn.LayerNorm(embed_dim*num_joints),
            nn.Linear(embed_dim*num_joints , 2),
        )
        self.frame_embed = nn.Parameter(torch.zeros(1, num_frame,1, out_dim))
        
        
        self.k_Project = nn.Linear(out_dim+embed_dim, embed_dim//2)
        self.v_Project = nn.Linear(out_dim+embed_dim, embed_dim//2)
        self.q_Project = nn.Linear(out_dim, embed_dim//2)
        
        self.FCFuser = nn.Linear(embed_dim*3, embed_dim//2)
        self.FCAct = nn.GELU()
        self.queryNorm = norm_layer(embed_dim//2)
        
        

    def TTE_foward(self, x):
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x, fineX, trendX = blk(x)
        # x = checkpoint(blk, x)  # Use checkpointing to save memory

        x = self.Temporal_norm(x)
        return x, fineX, trendX


    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        
        fineXs = []
        trendXs = []

        for i in range(1, self.block_depth):

            tteblock = self.TTEblocks[i]

            xT,fineX, trendX = tteblock(x)
            # x = checkpoint(tteblock, x)  # Use checkpointing to save memory
            xT = self.Temporal_norm(xT)
            fineXs.append(fineX)
            trendXs.append(trendX)
            
            x = xT
        
        return x, fineXs, trendXs

    def make_query(self, chosen_frames, last_frame, mask_valid=None, attn_drop=0, out_drop=0):
        K = self.k_Project(chosen_frames)   # [B, k, n, d_k]
        # V = self.v_Project(chosen_frames)   # [B, k, n, d_v]
        _, _, _, d_k = K.shape
        # last_frame -> Q
        Q = self.q_Project(last_frame)  # [B, 1, n, d_q]
        
        QK= torch.cat([Q.expand_as(K),K],dim=-1)
        QK = rearrange(QK,'b k n d -> b n (k d)').unsqueeze(1)  # [B, 1, n, k*d]
        query_out = self.FCFuser(QK)  # [B, 1, n, d_v]
        query_out = self.queryNorm(query_out)
        query_out = self.FCAct(query_out)
        query_out=query_out+Q

        return query_out

    def forward(self, x, with_entropy=False):
        b, f, n, c = x.shape
        
        x = self.Spatial_patch_to_embedding(x)
        
        xT, fineX, trendX = self.TTE_foward(x)
        x=xT
        
        x, fineXs, trendXs = self.ST_foward(x)
        y = self.head(x)
        y = y.view(b, f, n, -1)
        fineXs.insert(0, fineX)
        trendXs.insert(0, trendX)
        fineXs = torch.stack(fineXs, dim=1).squeeze(0)  # [B, depth, F, N, D]
        trendXs = torch.stack(trendXs, dim=1).squeeze(0)  # [B, depth, F, N, D]

        if with_entropy:
            return y, None, x, None
        else:
            return y, [fineXs,trendXs], x   # [b, f, n, out], [b, top_k, n, d], [b, f, n, d]

