import torch
from math import sqrt
from .utils.MambaUtil import *

class MambaNetSelectUni(torch.nn.Module):

    def __init__(self,):
        super(MambaNetSelectUni, self).__init__()
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
        self.register_buffer("last3y", torch.zeros(3, 5, 3))
        self.register_buffer("last3z", torch.zeros(3, 5, 128))


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

    def step(self,x, counter):
        y,z = self.model.step(x, counter)
        if counter == 199:
            self.last3y[-1]=y.clone()
            self.last3z[-1]=z.clone()
        elif counter == 198:
            self.last3y[-2]=y.clone()
            self.last3z[-2]=z.clone()
        elif counter == 197:
            self.last3y[-3]=y.clone()
            self.last3z[-3]=z.clone()
        return y
    
    def makeQuery(self):
        selected_frames = torch.cat((self.last3y, self.last3z), dim=-1).unsqueeze(0)
        last_frame = self.last3y[-1].unsqueeze(0)
        query_out = self.model.make_query(selected_frames, last_frame)
        return query_out

    def restWindow(self):
        self.model.resetBuf()