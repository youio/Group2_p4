
with open('../eval/stamped_traj_estimate.txt', 'r') as file:
    stamp_traj = float(file.readline().split()[0])
print(stamp_traj)

with open('../eval/stamped_groundtruth.txt', 'r') as file:
    stamp_gt = float(file.readline().split()[0])
print(stamp_gt)

print(stamp_traj-stamp_gt)