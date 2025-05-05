import torch
import numpy as np
import cv2
import argparse
import random
from Network import *
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(batchnum, mode):
    gt_traj = torch.tensor(np.load(f'.\\train\\traj{batchnum}.npy')).to(DEVICE, dtype=torch.float)
    imu_dat = None
    images = None
    imdir = f'.\\train\\imgs{batchnum}\\'

    if 'i' in mode:
        imu_dat = torch.tensor(np.load(f'.\\train\\imu{batchnum}.npy')[:,1:]).to(DEVICE, dtype=torch.float)
        imu_dat = imu_dat.reshape((100,10,6))
    if 'v' in mode:
        imprev = torch.from_numpy(cv2.imread(f'.\\train\\imgs{batchnum}\\0.png')).permute(2, 0, 1)
        images = torch.zeros((100,6, 224, 224)).to(DEVICE, dtype=torch.float)
        files = os.listdir(f'.\\train\\imgs{batchnum}')
        for i in range(1, len(files)-1):
            im = torch.from_numpy(cv2.imread(f'.\\train\\imgs{batchnum}\\{i*10}.png')).permute(2, 0, 1)
            images[i] = torch.concatenate([imprev, im], dim=0)
            imprev = im
            # print(im.shape)

        images = images.reshape(10, 10, 6, 224, 224)

    # gt_diff = gt_traj[1:] - gt_traj[:-1]
    gt_pos = gt_traj[:, 1:4]


    return imu_dat, images, gt_pos

def eulers_to_rots(angles):
    sinz = torch.sin(angles[:, 2])
    siny = torch.sin(angles[:, 1])
    sinx = torch.sin(angles[:, 0])

    cosz = torch.cos(angles[:, 2])
    cosy = torch.cos(angles[:, 1])
    cosx = torch.cos(angles[:, 0])

    zeros = torch.zeros((angles.shape[0])).to(DEVICE)
    ones = torch.ones((angles.shape[0])).to(DEVICE)

    x_rot = torch.concat([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=0)
    y_rot = torch.concat([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=0)
    z_rot = torch.concat([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=0)

    x_mats = x_rot.T.reshape(-1, 3, 3)
    y_mats = y_rot.T.reshape(-1, 3, 3)
    z_mats = z_rot.T.reshape(-1, 3, 3)

    rot_mats = x_mats @ y_mats @ z_mats

    return rot_mats



def get_traj_from_diffs(pos, orient):
    cum_orients = torch.cumsum(orient, dim=0)
    rot_mats = eulers_to_rots(cum_orients)

    pos_rotated = rot_mats @ pos.unsqueeze(2)

    pos_rotated = pos_rotated.squeeze(2)

    cum_pos = torch.cumsum(pos_rotated, dim=0)

    return cum_pos, cum_orients


def lossfn(pos, orient, gt_traj):

    # convert deltas to continuous trajectory
    pred_traj, _ = get_traj_from_diffs(pos, orient)
    # pred_traj = torch.cumsum(pos, dim=0)
    # implement rmse

    return torch.mean(torch.sqrt(torch.sum((gt_traj[9:10:1000]-pred_traj-gt_traj[0])**2, dim=1)))


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_data_path", default="./train", help="Base directory containing trajectory folders")
    parser.add_argument("--run_name", default="IO_bs_raft", help="Specific run log directory name (e.g., IO_bs_raft)")
    parser.add_argument("--flag", default="io", choices=['io', 'vo', 'vio'], help="Type of Odometry (io, vo, vio)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()

    # initialize model
    flag = args.flag
    if flag == 'vio':
        model = VIOmodel().to(DEVICE)
    elif flag == 'vo':
        model = VOmodel().to(DEVICE)
    elif flag == 'io':
        model = IOmodel().to(DEVICE)
    else:
        raise ValueError(f"Invalid flag: {flag}")

    
    # Initialize storage arrays
    train_acc = []
    val_acc = []
    best_loss = 1e8
    numbatches = 40
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        # Get random traj order
        traj_nums = list(range(40))
        random.shuffle(traj_nums)

        # Add scheduler and optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        print(f'Starting training for {flag} model.')
        for i, traj_num in enumerate(traj_nums):
            imu_dat, images, gt_traj = generate_batch(traj_num, flag)
            pos, orient = model(images, imu_dat)

            # implement
            pos, orient = model(images, imu_dat)
            loss = lossfn(pos, orient, gt_traj)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            print(f'  Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{40}], Loss: {loss.item():.4f}')

            # TODO: validation



            
        epoch_train_loss /= numbatches
        train_acc.append(epoch_train_loss)
        
        if epoch_train_loss < best_loss:
            print(f'new best loss: {epoch_train_loss} Epoch {epoch}')
            torch.save(model.state_dict(), f'.\\logs\\checkpoints\\{flag}_best_model.pt')
            best_loss = epoch_train_loss









