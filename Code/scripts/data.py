import csv
import numpy as np
from scipy.spatial.transform import Rotation as R


with open('../../mav0/state_groundtruth_estimate0/data.csv') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    init_pos = np.array([4.688319, -1.786938, 0.783338]) 
    init_orient = np.array([0.534108, -0.153020, -0.827383, -0.082152])
    rot = R.from_quat(init_orient).inv()

    f = open('../eval/stamped_groundtruth.txt', 'w')
    for row in csvreader:
        ts, tx, ty, tz, qw, qx, qy, qz = row[:8]
        r = (rot * R.from_quat([qx, qy, qz, qw])).as_quat().astype(str).tolist()
        t = (np.array([tx, ty, tz]).astype(float) - init_pos).astype(str).tolist()
        newrow = " ".join([ts+'e-9', t[0], t[1], t[2], r[0], r[1], r[2], r[3]])
        f.write(newrow+'\n')

    f.close()



