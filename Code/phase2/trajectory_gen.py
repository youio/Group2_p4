import numpy as np
from utils import *
from math import sin, cos, pi
from ImuUtils import run_acc_demo, run_gyro_demo


class Trajectory():
    def __init__(self, p_func=None, ndim=3, nchannels=2 ):
        if p_func == None:
            self.p_func = self.random_traj
            self.nchannels = nchannels
            self.ndim = ndim
            ais = np.random.rand(nchannels*ndim)*1.5+0.5
            bis = np.random.rand(nchannels*ndim)*9+1
            cis = np.random.rand(nchannels*ndim)*2*pi

            self.coefs = np.vstack([ais,bis,cis]).T.reshape(ndim,nchannels,-1)
        else:
            self.p_func = p_func
        
    def sample(self, t):
        samples = np.zeros((len(t), 3))
        for i, t in enumerate(t):
            # print(self.p_func(t))
            samples[i, :] = self.p_func(t)
        
        return samples
    
    def sample_at(self, t):
        return self.p_func(t)
    
    def random_traj(self, t):
        
        res = []

        # assign xy coordinates
        for i in range(self.ndim):
            thisdim = self.coefs[i].T
            thisdim_res = np.sum(thisdim[0]*np.sin(thisdim[1]*t+thisdim[2]))

            res.append(thisdim_res)

        # # asign z coordinate
        # x, y = res
        # # res.append((x*cos(self.z_coefs[1])+y*sin(self.z_coefs[1]))*sin(2*t+self.z_coefs[0]))
        # res.append(0)
        return res
    
    def get_orients_heading(trajectory):
        base = trajectory[1]-trajectory[0]
        base[2] = 0
        orients = [from_two_vectors(base)]

        for i in range(1, len(trajectory)):
            displace_vec = trajectory[i]-trajectory[i-1]
            displace_vec[2] = 0
            orient = from_two_vectors(base, displace_vec)

            orients.append(orient)

        return orients
    
    
    
def write_trajectory_and_imu(orients, trajectory, imu_data, traj_dest, imu_dest):
    
    # normalize timescale from 0-10
    timestamps = np.linspace(0,10,1000).reshape(-1,1)

    # print(trajectory.shape, orients.shape, timestamps.shape)
    pose_traj = np.hstack([timestamps, trajectory, orients])
    print(pose_traj)
    # print(pose_traj, pose_traj.shape)
    imu_stamped = np.hstack([timestamps, imu_data])

    np.save(traj_dest, pose_traj)
    np.save(imu_dest, imu_stamped)

    return pose_traj


def get_imu_ref(trajectory, orients, dt):
    linvel = descrete_derivative(trajectory, dt)
    linacc = descrete_derivative(linvel, dt)
    linacc = np.vstack([linacc[0],linacc[0], linacc])
    angvel = descrete_derivative(orients, dt)
    angvel = np.vstack([angvel[0], angvel])

    return linacc, angvel


def generate_random_traj(t, ndim=3, nchannels=2):

    # Define random coefficients
    ais = np.random.rand(nchannels*ndim)*1.5+0.5
    bis = np.random.rand(nchannels*ndim)*9+1
    cis = np.random.rand(nchannels*ndim)*2*pi

    coefs = np.vstack([ais,bis,cis]).T.reshape(ndim,nchannels,-1)
    print(coefs)

    # generate trajectory
    samples = np.zeros((len(t), 3))
    for j, this_t in enumerate(t):
        this_res = []
        for i in range(ndim):
            thisdim = coefs[i].T
            thisdim_res = np.sum(thisdim[0]*np.sin(thisdim[1]*this_t+thisdim[2]))

            this_res.append(thisdim_res)
        # print(thiss_res)
        samples[j, :] = this_res    
    return samples

def get_random_orients(trajectory, t):

    coefs = np.random.rand(6)*pi

    random_ang_vel = coefs[0]*np.sin(coefs[1]*t+coefs[2]) + coefs[3]*np.sin(coefs[4]*t+coefs[5])
    random_ang_vel = (random_ang_vel/np.max(random_ang_vel))*0.001
    
    base = trajectory[1]-trajectory[0]
    base[2] = 0
    orients = [[0,0,0]]

    for i in range(len(t)-1):
        # rotate  orientation by small angle scaled by the random angular velocity series
        # rotated = R.from_quat(np.array([ 0, 0, sin(0.001/2),cos(0.001/2)])*random_ang_vel[i])*R.from_quat(orients[i])
        # orient = rotated.as_quat()
        orient = orients[-1].copy()
        orient[2] += random_ang_vel[i]
        # print(orients)
        orients.append(orient)

    return np.array(orients)

def generate_num_data(i, nsamp=1000):
    t = np.linspace(0, pi/3, nsamp)
    traj = generate_random_traj(t)
    orients = get_random_orients(traj, t)

    linacc_ref, angvel_ref = get_imu_ref(traj, orients, 10/nsamp)
    linacc_real = run_acc_demo(linacc_ref)
    angvel_real = run_gyro_demo(angvel_ref)

    time_real = np.linspace(0, 10, nsamp)

    imu_dat = np.concatenate([linacc_real, angvel_real], 1)

    pose_dat = write_trajectory_and_imu(orients, traj, imu_dat, f'./train/traj{i}.npy', f'./train/imu{i}.npy')
    
    
    # plot_trajectory(traj)
    plot_imu_data(linacc_real, time_real)
    plot_imu_data(angvel_real, time_real)
    
    
    return pose_dat
    # print(imu_dat.shape)

    




if __name__ == '__main__':
    # trefoil = Trajectory(lambda t: [0.5*sin(t)+sin(2*t), 0.5*cos(t)-cos(2*t), -0.5*sin(3*t)])
    # fig8 = Trajectory(lambda t : [sin(t), sin(2*t), 1])
    # ellipse = Trajectory(lambda t : [2*cos(t), sin(t), 1])
    # spiral = Trajectory(lambda t : [cos(t), sin(t), t/pi+0.5])

    # random_traj.plot(0, pi/2)
    ntraj = 1
    for i in range(ntraj):
        generate_num_data(i)

    # random_traj.write_trajectory_and_imu(0, pi/2, './train/testgen.npy','./train/testimu.npy' )



# heart = Trajectory(lambda t : [16*sin(t)**3, 13*cos(t)- 5*cos(2*t)-2*cos(3*t)-cos(4*t), sin(0.5*t)])

# heart.plot(0, 2*pi)