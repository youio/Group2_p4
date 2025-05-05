import torch
from Train import *
from Network import *
from trajectory_gen import plot_trajectories
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_trajectories(model, mode):
    model.eval()
    for i in range(10):
        imu_dat, images, gt_traj = generate_batch(i, mode)

        pos, orients = model(images, imu_dat)

        traj, orients = get_traj_from_diffs(pos, orients)

        # timestamps = np.linspace(0,10,1000).reshape(-1,1)

        # pose_traj = np.hstack([timestamps, traj, orients])

        plot_trajectories([traj.cpu().detach(), gt_traj.cpu()])
        # np.save(pose_traj, f'./logs/pred_traj{i}.npy')

    

if __name__ == '__main__':
    
    vo = VOmodel().to(DEVICE)
    # vo = VOmodel().to(DEVICE)
    # io = IOmodel().to(DEVICE)

    vo.load_state_dict(torch.load('.\\logs\\checkpoints\\vo_best_model.pt'))

    write_trajectories(vo, 'vo')



