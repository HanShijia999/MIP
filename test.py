import torch
import tqdm
import articulate as art
from net import DuMambaUniSelect


device= torch.device('cuda:0')  # Force CPU for testing


class ReducedPoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator('models/SMPL_male.pkl', joint_mask=torch.tensor([1, 2, 16, 17]), device=device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def __call__(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 1000])


class FullPoseEvaluator:
    names = ['Absolute Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator('models/SMPL_male.pkl', device=device)

    def __call__(self, pose_p, pose_t):
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[4] / 1000])


def compare_realimu(data, dataset_name='', evaluate_pose=True):
    print('======================= Testing on %s Real Dataset =======================' % dataset_name)
    reduced_pose_evaluator = ReducedPoseEvaluator()
    full_pose_evaluator = FullPoseEvaluator()
    g = torch.tensor([0, -9.8, 0])
    batch_nets={
        'MIP':DuMambaUniSelect(device=device).eval().to(device),
    }
    pose_errors = {k: [] for k in batch_nets.keys()}
    for k in batch_nets:
        pose_errors[k]=[]


    for seq_idx in range(len(data['pose'])):

        aS = data['aS'][seq_idx]
        wS = data['wS'][seq_idx]
        mS = data['mS'][seq_idx]
        RIS = data['RIS'][seq_idx]
        RIM = data['RIM'][seq_idx]
        RSB = data['RSB'][seq_idx]
        tran = data['tran'][seq_idx]
        pose = data['pose'][seq_idx]

        RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB).to(device)
        aM = (RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + g).to(device)
        wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1).to(device)
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)


        step=200
        # pose=pose[:step]
        # tran=tran[:step]

        for net in batch_nets.values():
            net.rnn_initialize(pose[0])
            net.pose_prediction = torch.zeros_like(pose)

        
        for i in tqdm.trange(0,pose.shape[0],step):
            for net in batch_nets.values():
                try:
                    net.pose_prediction[i:i+step] = net.forward_batch(aM[i:i+step], wM[i:i+step], RMB[i:i+step])
                except AssertionError:
                    left_len=len(aM)-i
                    pad_len=step-(left_len)

                    aM_pad = torch.cat([aM[i:], aM[-1:].repeat(pad_len, 1, 1)], dim=0)
                    wM_pad = torch.cat([wM[i:], wM[-1:].repeat(pad_len, 1, 1)], dim=0)
                    RMB_pad = torch.cat([RMB[i:], RMB[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
                    pose_pad = net.forward_batch(aM_pad, wM_pad, RMB_pad)
                    net.pose_prediction[i:] = pose_pad[:left_len]


        if evaluate_pose:
            print('[%3d/%3d  pose]' % (seq_idx, len(data['pose'])), end='')
            for k in batch_nets.keys():
                e1 = reduced_pose_evaluator(batch_nets[k].pose_prediction, pose, )
                e2 = full_pose_evaluator(batch_nets[k].pose_prediction, pose,)
                pose_errors[k].append(torch.cat((e1, e2), dim=0))
                print('\t%s: %5.2fcm' % (k, pose_errors[k][-1][2, 0]), end=' ')  # joint position error
            print('')

    print('======================= Results on %s Real Dataset =======================' % dataset_name)
    if evaluate_pose:
        print('Metrics: ', reduced_pose_evaluator.names + full_pose_evaluator.names)
        for net_name, error in pose_errors.items():
            error = torch.stack(error).mean(dim=0)
            print(net_name, end='\t')
            for error_item in error:
                print('%.2f±%.2f' % (error_item[0], error_item[1]), end='\t')  # mean & std
            print('')
    

def compare_realimu_steps(data, dataset_name='', evaluate_pose=True, evaluate_tran=False):
    print('======================= Testing on %s Real Dataset =======================' % dataset_name)
    reduced_pose_evaluator = ReducedPoseEvaluator()
    full_pose_evaluator = FullPoseEvaluator()
    g = torch.tensor([0, -9.8, 0])
    nets = {
        'MPI':DuMambaUniSelect(device=device).eval().to(device),
        # 'PNP (Trans)':PNPT().eval().to(device),
        # 'PNP (Mamba)':PNPM(weight_file='weightsMamba.pt',positional=True).eval().to(device),
        # 'PNP (IKtransform)':PNPTIK().eval().to(device)
    }
    pose_errors = {k: [] for k in nets.keys()}

    for seq_idx in range(len(data['pose'])):
        aS = data['aS'][seq_idx]
        wS = data['wS'][seq_idx]
        mS = data['mS'][seq_idx]
        RIS = data['RIS'][seq_idx]
        RIM = data['RIM'][seq_idx]
        RSB = data['RSB'][seq_idx]
        tran = data['tran'][seq_idx]
        pose = data['pose'][seq_idx]

        RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB).to(device)
        aM = (RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + g).to(device)
        wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1).to(device)
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        for net in nets.values():
            net.rnn_initialize(pose[0])
            net.pose_prediction = torch.zeros_like(pose)

        for i in tqdm.trange(pose.shape[0]):
            for net in nets.values():
                net.pose_prediction[i] = net.forward_frame(aM[i], wM[i], RMB[i])





        if evaluate_pose:
            print('[%3d/%3d  pose]' % (seq_idx, len(data['pose'])), end='')
            for k in nets.keys():
                e1 = reduced_pose_evaluator(nets[k].pose_prediction, pose, )
                e2 = full_pose_evaluator(nets[k].pose_prediction, pose,)
                pose_errors[k].append(torch.cat((e1, e2), dim=0))
                print('\t%s: %5.2fcm' % (k, pose_errors[k][-1][2, 0]), end=' ')  # joint position error
            print('')

    print('======================= Results on %s Real Dataset =======================' % dataset_name)
    if evaluate_pose:
        print('Metrics: ', reduced_pose_evaluator.names + full_pose_evaluator.names)
        for net_name, error in pose_errors.items():
            error = torch.stack(error).mean(dim=0)
            print(net_name, end='\t')
            for error_item in error:
                print('%.2f±%.2f' % (error_item[0], error_item[1]), end='\t')  # mean & std
            print('')
    


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    data = torch.load('data/test_datasets/dipimu.pt')
    # compare_realimu(data, dataset_name='DIP_IMU')
    compare_realimu_steps(data, dataset_name='DIP_IMU')
