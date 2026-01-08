import torch
import articulate as art
from articulate.utils.torch.rnn import *
from dynamics import PhysicsOptimizer


class NumericalDifferentiation:
    def __init__(self, dt=1., n_points=2):
        assert n_points in [2, 3, 5]
        self.dt = dt
        self.n_points = n_points
        self.reset()

    def reset(self):
        self.x = [None] * (self.n_points - 1)

    def __call__(self, x):
        if self.x[0] is None:
            d = torch.zeros_like(x)
        elif self.n_points == 2:
            d = (x - self.x[0]) / self.dt
        elif self.n_points == 3:
            d = (3 * x - 4 * self.x[1] + self.x[0]) / (2 * self.dt)
        elif self.n_points == 5:
            d = (25 * x - 48 * self.x[3] + 36 * self.x[2] - 16 * self.x[1] + 3 * self.x[0]) / (12 * self.dt)
        self.x = self.x[1:] + [x.clone()]
        return d


from NetBank.VanilaMamba import MambaNetSelectUni
class DuMambaUniSelect(torch.nn.Module):
    name = 'MIP'
    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ji_ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    def __init__(self, cheat=None, weight_file='weights.pt',mamba_weight='best_weights.pt',device='cpu'):
        super(DuMambaUniSelect, self).__init__()
        self.device = device
        self.cheat=cheat

        self.num_frame = 200
        self.model = MambaNetSelectUni()
        
        self.iknet_net1 = RNN(input_linear=False,
                              input_size=45 + 15,
                              output_size=69,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.iknet_net2 = RNN(input_linear=False,
                              input_size=45 + 69,
                              output_size=90,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.vrnet_net1 = RNN(input_linear=False,
                              input_size=144,
                              output_size=72,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.vrnet_net2 = RNN(input_linear=False,
                              input_size=144,
                              output_size=2,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)


        self.to(self.device).load_state_dict(torch.load('data/weights/PNP/'+weight_file, map_location=self.device),strict=False)
        self.model.load_state_dict(torch.load('NetBank/weights/'+mamba_weight,map_location=self.device), strict=False)
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.vi_mask, device=self.device)
        self.dynamics_optimizer = PhysicsOptimizer(debug=False, quiet=False)
        self.nd = NumericalDifferentiation(dt=1/60)
        self.nd2 = NumericalDifferentiation(dt=1/60)
        self.rnn_initialize()  # using T-pose
        self.eval()

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None):
        if init_pose is None:
            init_pose = torch.eye(3, device=self.device).expand(1, 24, 3, 3)
        else:
            init_pose = init_pose.view(1, 24, 3, 3).to(self.device)
            init_pose[0, 0] = torch.eye(3, device=self.device)
        pl = self.body_model.forward_kinematics(init_pose, calc_mesh=True)[2].view(6, 3)
        pl = (pl[:5] - pl[5:]).ravel()
        # self.pl1hc = [_.contiguous() for _ in self.plnet_net1.init_net(pl).view(1, 2, self.plnet_net1.num_layers, self.plnet_net1.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.vr1hc = None
        self.vr2hc = None
        # self.pRB_dyn = pl.view(5, 3, 1)
        self.pRB_dyn = self.model.initQueryNet(pl.view(5, 3))#[1,5,64]

        self.dynamics_optimizer.reset_states()
        self.nd.reset()
        self.nd2.reset()
        self.buffer=[]


    @torch.no_grad()
    def forward_batch(self, a, w, R):
        N= a.shape[0]
        a = a.reshape(N,  6, 3, 1)
        w = w.reshape(N, 6, 3, 1)
        R = R.reshape(N, 6, 3, 3)

        RIR = R[:, 5:]
        aRB_sta = RIR.transpose(-2, -1).matmul(a)
        wRB_sta = RIR.transpose(-2, -1).matmul(w)
        RRB_sta = RIR.transpose(-2, -1).matmul(R)
        wRR_sta = wRB_sta[:, 5:]
        wRB_sta = wRB_sta[:, :5]
        RRB_dyn = RRB_sta[:,:5]
        aRB_dyn = aRB_sta[:, :5] - aRB_sta[:, 5:]#agt
        
        pRB_dyn = self.pRB_dyn.view(1, 5, -1).expand(N, 5, -1)

        x1 = torch.cat((aRB_dyn.flatten(2) / 20, RRB_dyn.flatten(2), wRB_sta.flatten(2), pRB_dyn.flatten(2)), dim=2).unsqueeze(0) #[1,seq, 5, 18]
        out_chunk, nextQuery,  = self.model(x1) # [1, seq, 5, 3]
        self.pRB_dyn=nextQuery.squeeze(0).squeeze(0)
        

        x = torch.cat((RRB_dyn.view(N, -1), out_chunk.view(N,-1)), dim=1).unsqueeze(1) #[seq,batch=1, 60]
        x, self.ik1hc = self.iknet_net1.rnn(x, self.ik1hc) #[seq, batch=1, 512]
        x = self.iknet_net1.linear2(x)#[T, 1, D=69]

        jpos = x.view(N, 1, 23, 3)
        # jpos may be the all joints position and etc
        # total SMPL joints are 24
        # one joint is missing.
        # IK-s2
        x = torch.cat((RRB_dyn.view(N, 1, -1), jpos.flatten(-2)), dim=-1)
        x, self.ik2hc = self.iknet_net2.rnn(x, self.ik2hc)
        x = self.iknet_net2.linear2(x) #[T, 1, D=90]
        # may be the orientations of all joints
        # x.shape = [1, 90], meaning [1, 15, 6]
        # then use r6d_to_rotation_matrix() to make it to [1, 15, 3, 3]
        # subset of 24 joints


        # get pose estimation
        reduced_glb_pose = art.math.r6d_to_rotation_matrix(x).view(N, 15, 3, 3)
        glb_pose = torch.eye(3, device=self.device).repeat(N, 24, 1, 1)
        glb_pose[:, self.ji_reduced] = reduced_glb_pose
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(N, 24, 3, 3)
        pose[:, self.ji_ignored] = torch.eye(3, device=self.device)

        pose[:,:1] = RIR

        joint = self.body_model.forward_kinematics(pose.view(N, 24, 3, 3).to(self.device))[1].view(N,24, 3)
        # the joint position calculated with rotations
        aj = joint[:,1:]@(RIR.view(N,3,3))
        # aj     : location of joints in root frame
        # aRB_sta: raw acceleration in root frame
        # RRB_sta: raw rotations in root frame
        # wRR_sta: raw angular velocity in root frame
        imu = torch.cat((aRB_sta.flatten(1) / 20, RRB_sta.flatten(1), wRR_sta.flatten(1) / 4, aj.flatten(1)),dim=1)
        
        # VR-s1
        x, self.vr1hc = self.vrnet_net1.rnn(imu.unsqueeze(1), self.vr1hc)
        x = self.vrnet_net1.linear2(x.squeeze(1))
        av = x.view(N,24, 3) * 2
        # velocities in root frame
        # global motion estimator

        # VR-s2
        x, self.vr2hc = self.vrnet_net2.rnn(imu.unsqueeze(1), self.vr2hc)
        x = self.vrnet_net2.linear2(x.squeeze(1))
        c = x.view(N,2)
        # ground contact probability

        # physics-based optimization
        av =av@RIR.squeeze(1).transpose(-2,-1)
        # pose: smpl pose
        # av: all joint velocities in global frame
        # c: ground contact probability
        # a: raw acceleration from IMU in global frame
        # pose_opt, tran_opt = self.dynamics_optimizer.optimize_frame(pose.cpu(), av.cpu(), c.cpu(), a.cpu(), return_grf=False)
        # self.pRB_dyn=self.forward_kinematics(pose_opt, tran_opt)
        pose_opt=[]
        T = pose.shape[0]
        a=a.squeeze(-1)
        for i in range(T):
            # print(pose.shape,av.shape,c.shape,a.shape)
            #torch.Size([24, 3, 3]) torch.Size([24, 3]) torch.Size([2]) torch.Size([6, 3])
            _pose_opt, _ = self.dynamics_optimizer.optimize_frame(pose.cpu()[i], av.cpu()[i], c.cpu()[i], a.cpu()[i], return_grf=False)
            pose_opt.append(_pose_opt)
        pose_opt = torch.stack(pose_opt,dim=0)
        return pose_opt



