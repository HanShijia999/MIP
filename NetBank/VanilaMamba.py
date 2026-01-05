import torch
from math import sqrt
from .utils.PoseMambaUtil import *

class VanilaMambaNetSelectUni(torch.nn.Module):

    def __init__(self,):
        super(VanilaMambaNetSelectUni, self).__init__()
        self.num_frame=200
        
        self.model = VanilaMambaSelectAttenPatchWithLayerOutput(num_frame=self.num_frame, 
                               num_joints = 5,
                               in_chans=15+128//2,
                               embed_dim_ratio=128, 
                               mlp_ratio = 2, 
                               depth =15)
        self.initQueryNet = torch.nn.Sequential(
            torch.nn.Linear(3, 64*2),
            torch.nn.ReLU(),
            torch.nn.Linear(64*2, 64*2),
            torch.nn.ReLU(),
            torch.nn.Linear(64*2, 128//2),
        )


    def forward(self, x):
        y, _, z = self.model(x)
        
        N = y.shape[1]
        query_out=None
        if N>3:
            last_y = y[:,-3:]
            last_z = z[:,-3:]
            
            selected_frames = torch.cat((last_y, last_z), dim=-1)# [B, k, 5, 3+d]
            last_frame = y[:,-1]
            
            query_out = self.model.make_query(selected_frames, last_frame)
        return  y, query_out
