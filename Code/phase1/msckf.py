import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple
from scipy.spatial.transform import Rotation
from scipy import sparse




class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81])

    # Transformation offset from the IMU frame to the body frame. 
    # The transformation takes a vector from the IMU frame to the 
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([-0.153029, -0.827383,	-0.082152, 0.534108])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)

        
class StateServer(object):
    """
    Store one IMU states and several camera states for constructing 
    measurement model.
    """
    def __init__(self):
        self.imu_state = IMUState()
        self.cam_states = dict()   # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self.state_cov = np.zeros((21, 21))
        self.continuous_noise_cov = np.zeros((12, 12))



class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()   # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        # The intial orientation and position will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3],
            config.T_imu_body[:3, 3])

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will 
        # start after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed 
        # immediately. The IMU msgs are processed when the next image is  
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        # Propogate the IMU state.
        # that are received before the image msg.
        self.batch_imu_processing(feature_msg.timestamp)

        print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print('---prune_cam_state_buffer  ', time.time() - t)
        print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

    def initialize_gravity_and_bias(self):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Initialize the IMU bias and initial orientation based on the 
        first few IMU readings.
        """
        sum_angular_vel = np.zeros(3)
        sum_linear_acc = np.zeros(3)

        # Finding gravity in the IMU frame.
        for imu_msg in self.imu_msg_buffer:

            sum_angular_vel += imu_msg.angular_velocity
            sum_linear_acc += imu_msg.linear_acceleration

        self.state_server.imu_state.gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        gravity_imu = sum_linear_acc / len(self.imu_msg_buffer)
        gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = gyro_bias

        # Normalizing gravity and save to IMUState
        gravity_norm = np.linalg.norm(gravity_imu)
        IMUState.gravity = np.array([0.0, 0.0, -gravity_norm])


        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        IMUState.gravity = np.array([0., 0., -gravity_norm])

        self.state_server.imu_state.orientation = from_two_vectors(-IMUState.gravity, gravity_imu)

    def batch_imu_processing(self, time_bound):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Process the imu message given the time bound
        """
        used_imu_msg_counter = 0
        for imu_msg in self.imu_msg_buffer:
            imu_time = imu_msg.timestamp
            if imu_time < self.state_server.imu_state.timestamp:
                used_imu_msg_counter += 1
                continue
            # Break if the time_bound is reached
            if imu_time > time_bound:
                break
            # Execute process model
            self.process_model(imu_time, imu_msg.angular_velocity, imu_msg.linear_acceleration)
            used_imu_msg_counter += 1
            # Update the state info
            self.state_server.imu_state.timestamp = imu_time
        
        # Set the current imu id to be the IMUState.next_id
        self.state_server.imu_state.id = IMUState.next_id
        
        # IMUState.next_id increments
        IMUState.next_id += 1

        # Remove all used IMU msgs.
        del self.imu_msg_buffer[:used_imu_msg_counter]

    def process_model(self, time, m_gyro, m_acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.A: The dynamics of the error IMU state following equation (2) in the "MSCKF" paper.
        """
        # Get the error IMU state
        imu_state = self.state_server.imu_state
        gyro = m_gyro - imu_state.gyro_bias
        acc = m_acc - imu_state.acc_bias
        dt = time - imu_state.timestamp

        # Compute discrete transition F, Q matrices in Appendix A in "MSCKF" paper
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))
        
        F[:3, :3] = -skew(gyro)
        F[:3, 3:6] = -np.eye(3)
        F[6:9, :3] = -to_rotation(imu_state.orientation).T @ skew(acc)
        F[6:9, 9:12] = -to_rotation(imu_state.orientation).T
        F[12:15, 6:9] = np.eye(3)

        # Block assignments for G
        G[:3, :3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)
        G[6:9, 6:9] = -to_rotation(imu_state.orientation).T
        G[9:12, 9:12] = np.eye(3)
        
        # Approximate matrix exponential to the 3rd order, which can be 
        # considered to be accurate enough assuming dt is within 0.01s.
        Fdt = F * dt
        Fdt2 = Fdt @ Fdt
        Fdt3 = Fdt2 @ Fdt
        Phi = np.eye(21) + Fdt + 0.5 * Fdt2 + (1/6) * Fdt3

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(dt, gyro, acc)

        # Modify the transition matrix
        R_kk_1 = to_rotation(imu_state.orientation_null)
        Phi[:3, :3] = to_rotation(imu_state.orientation) @ R_kk_1.T
        
        u = (R_kk_1 @ IMUState.gravity).reshape(-1, 1)
        s = np.linalg.inv(u.T @ u) @ u.T
        # s = u / (u @ u)
        A1 = Phi[6:9, :3]
        print()
        w1 = skew(imu_state.velocity_null - imu_state.velocity) @ imu_state.gravity
        Phi[6:9, :3] = A1 - (A1@u - w1) * s
        A2 = Phi[12:15, :3]
        w2 = skew(dt*imu_state.velocity_null + imu_state.position_null - imu_state.position) @ imu_state.gravity
        Phi[12:15, :3] = A2 - (A2@u - w2) * s
        
        # Propogate the state covariance matrix.
        Q = Phi @ G @ self.state_server.continuous_noise_cov @ G.T @ Phi.T * dt
        self.state_server.state_cov[:21, :21] = Phi @ self.state_server.state_cov[:21, :21] @ Phi.T + Q

        if len(self.state_server.cam_states) > 0:
            # Propogate the camera states
            self.state_server.state_cov[:21, 21:] = Phi @ self.state_server.state_cov[:21, 21:]
            self.state_server.state_cov[21:, :21] = self.state_server.state_cov[21:, :21] @ Phi.T
            
         
        # Fix the covariance to be symmetric
        self.state_server.state_cov = 0.5 * (self.state_server.state_cov + self.state_server.state_cov.T)
        
        # Update the state correspondes to null space.
        imu_state.orientation_null = imu_state.orientation
        imu_state.position_null = imu_state.position
        imu_state.velocity_null = imu_state.velocity
        imu_state.timestamp = time


        print(f"Done process_model at {time}")
        

    def predict_new_state(self, dt, gyro, acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """Propogate the state using 4th order Runge-Kutta for equstion (1) in "MSCKF" paper"""
        norm_gyro = np.linalg.norm(gyro)
        
        # Get the Omega matrix, the equation above equation (2) in "MSCKF" paper
        Omega = np.zeros((4, 4))
        Omega[:3, :3] = -skew(gyro)
        Omega[:3, 3] = gyro
        Omega[3, :3] = -gyro
        
        # Get the orientation, velocity, position
        q = self.state_server.imu_state.orientation
        v = self.state_server.imu_state.velocity
        p = self.state_server.imu_state.position
        
        # Compute the dq_dt, dq_dt2 in equation (1) in "MSCKF" paper
        if norm_gyro > 1e-5: # almost zero
            dq_dt = (np.cos(norm_gyro*dt/2) * np.eye(4) + 1/norm_gyro * np.sin(norm_gyro*dt/2) * Omega) @ q
            dq_dt2 = (np.cos(norm_gyro*dt/4) * np.eye(4) + 1/norm_gyro * (np.sin(norm_gyro*dt/4) * Omega)) @ q
        else:
            dq_dt = np.cos(norm_gyro*dt/2) * (np.eye(4) + dt/2 * Omega) @ q
            dq_dt2 = np.cos(norm_gyro*dt/4) * (np.eye(4) + dt/4 * Omega) @ q
            
        dR_dt_transpose = to_rotation(dq_dt).T
        dR_dt2_transpose = to_rotation(dq_dt2).T
        
        # Apply 4th order Runge-Kutta 
        # k1 = f(tn, yn)
        k1_v_dot = to_rotation(q).T @ acc + IMUState.gravity
        k1_p_dot = v
        

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k1_v = v + k1_v_dot*dt/2.
        k2_p_dot = k1_v
        k2_v_dot = dR_dt2_transpose @ acc + IMUState.gravity
        
        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k2_v = v + k2_v_dot*dt/2
        k3_p_dot = k2_v
        k3_v_dot = dR_dt2_transpose @ acc + IMUState.gravity
        
        # k4 = f(tn+dt, yn+k3*dt)
        k3_v = v + k3_v_dot*dt
        k4_p_dot = k3_v
        k4_v_dot = dR_dt_transpose @ acc + IMUState.gravity

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q = dq_dt / np.linalg.norm(dq_dt)
        v = v + (k1_v_dot + 2*k2_v_dot + 2*k3_v_dot + k4_v_dot)*dt/6
        p = p + (k1_p_dot + 2*k2_p_dot + 2*k3_p_dot + k4_p_dot)*dt/6

        # update the imu state
        self.state_server.imu_state.orientation = q
        self.state_server.imu_state.velocity = v
        self.state_server.imu_state.position = p

    
    def state_augmentation(self, time):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Compute the state covariance matrix in equation (3) in the "MSCKF" paper.
        """
        imu_state = self.state_server.imu_state
        R_i_c = imu_state.R_imu_cam0
        t_c_i = imu_state.t_cam0_imu

        # Add a new camera state to the state server.
        R_w_i = to_rotation(imu_state.orientation)
        R_w_c = R_i_c @ R_w_i
        t_c_w = imu_state.position + R_w_i.T @ t_c_i
        
        self.state_server.cam_states[imu_state.id] = CAMState(imu_state.id)
        self.state_server.cam_states[imu_state.id].timestamp = time
        self.state_server.cam_states[imu_state.id].orientation = to_quaternion(R_w_c)
        self.state_server.cam_states[imu_state.id].position = t_c_w
        self.state_server.cam_states[imu_state.id].orientation_null = to_quaternion(R_w_c)
        self.state_server.cam_states[imu_state.id].position_null = t_c_w

        # Update the covariance matrix of the state.
        # To simplify computation, the matrix J below is the nontrivial block
        # Appendix B of "MSCKF" paper.
        J = np.zeros((6, 21))
        J[:3, :3] = R_i_c
        J[:3, 15:18] = np.eye(3)
        J[3:6, :3] = skew(R_w_i.T @ t_c_i)
        J[3:6, 12:15] = np.eye(3)
        J[3:6, 18:21] = np.eye(3)

        # Resize the state covariance matrix.
        old_rows, old_cols = self.state_server.state_cov.shape
        new_cov = np.zeros((old_rows + 6, old_cols + 6))
        new_cov[:old_rows, :old_cols] = self.state_server.state_cov

        # Fill in the augmented state covariance.
        new_cov[old_rows:, :old_cols] = J @ new_cov[:21, :old_rows]
        new_cov[:old_rows, old_cols:] = new_cov[old_rows:, :old_cols].T
        new_cov[old_rows:, old_cols:] = J @ new_cov[:21, :21] @ J.T

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (new_cov + new_cov.T) / 2
        
        print(f"Done state_augmentation at {time}")

    def add_feature_observations(self, feature_msg):
        """
        IMPLEMENT THIS!!!!!
        """
        # get the current imu state id and number of current features
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0
        
        # add all features in the feature_msg to self.map_server
        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                new_feature = Feature(feature.id, self.optimization_config)
                new_feature.observations[state_id] = np.array([feature.u0, feature.v0, feature.u1, feature.v1])
                self.map_server[feature.id] = new_feature
            else: # update existing feature
                self.map_server[feature.id].observations[state_id] = np.array([feature.u0, feature.v0, feature.u1, feature.v1])
                tracked_feature_num += 1
            
        # update the tracking rate
        self.tracking_rate = tracked_feature_num / (curr_feature_num + 1e-5)

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = to_rotation(cam_state.orientation)
        t_c0_w = cam_state.position

        # Cam1 pose.
        R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc   # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg   # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x   # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2]/p_c0[2], *p_c1[:2]/p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id 
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states
        H_xj = np.zeros((jacobian_row_size, 
            21+len(self.state_server.cam_states)*6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count+4, 21+6*idx:21+6*(idx+1)] = H_xi
            H_fj[stack_count:stack_count+4, :3] = H_fi
            r_j[stack_count:stack_count+4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H, r):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.B: by stacking multiple observations, we can compute the residuals in equation (6) in "MSCKF" paper 
        """
        if H.shape[0] == 0 or r.shape[0] == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity.
        if H.shape[0] > H.shape[1]:
            # Perform QR decomposition on H
            Q, R = np.linalg.qr(H, mode='reduced')  # if M > N, return (M, N), (N, N)
            H_thin = R         # shape (N, N)
            r_thin = Q.T @ r   # shape (N,)
            
        else:
            H_thin = H
            r_thin = r

        # Compute the Kalman gain.
        P = self.state_server.state_cov
        S = H_thin @ P @ H_thin.T + (self.config.observation_noise * np.eye(len(H_thin)))
        K = np.linalg.solve(S, H_thin @ P).T

        # Compute the error of the state.
        delta_x = K @ r_thin
        
        # Update the IMU state.
        delta_x_imu = delta_x[:21]
        if np.linalg.norm(delta_x_imu[6:9]) > 0.5 or np.linalg.norm(delta_x_imu[12:15]) > 1:
            print('Update too large')
            
        dq_imu = small_angle_quaternion(delta_x_imu[:3])
        self.state_server.imu_state.orientation = quaternion_multiplication(dq_imu, self.state_server.imu_state.orientation)
        self.state_server.imu_state.gyro_bias += delta_x_imu[3:6]
        self.state_server.imu_state.velocity += delta_x_imu[6:9]
        self.state_server.imu_state.acc_bias += delta_x_imu[9:12]
        self.state_server.imu_state.position += delta_x_imu[12:15]
        
        dq_extrinsic = small_angle_quaternion(delta_x[15:18])
        self.state_server.imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) @ self.state_server.imu_state.R_imu_cam0
        self.state_server.imu_state.t_cam0_imu += delta_x[18:21]
        
        # Update the camera states.
        for i, cam_state in self.state_server.cam_states.items():
            idx = list(self.state_server.cam_states.keys()).index(i)
            delta_x_cam = delta_x[21+6*idx:21+6*(idx+1)]
            dq_cam = small_angle_quaternion(delta_x_cam[:3])
            cam_state.orientation = quaternion_multiplication(dq_cam, cam_state.orientation)
            cam_state.position += delta_x_cam[3:6]

        # Update state covariance.
        I_KH = np.eye(K.shape[0]) - K @ H_thin
        state_cov = I_KH @ P

        # Fix the covariance to be symmetric
        self.state_server.state_cov = 0.5 * (state_cov + state_cov.T)

    def gating_test(self, H, r, dof):
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1+P2, r)

        if(gamma < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available 
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += (4 * len(feature.observations) - 3)
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros((jacobian_row_size, 
            21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids)-1):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(
            cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(
                cam_state_pairs[cam_state_idx][1].orientation)
            
            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(
                rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids


    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4*len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6*idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        state_cov = np.zeros((21, 21))
        state_cov[ 3: 6,  3: 6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[ 6: 9,  6: 9] = self.config.velocity_cov * np.identity(3)
        state_cov[ 9:12,  9:12] = self.config.acc_bias_cov * np.identity(3)
        state_cov[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        state_cov[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_cam0 = self.state_server.imu_state.R_imu_cam0
        imu_state.t_cam0_imu = self.state_server.imu_state.t_cam0_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if 
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if max(position_x_std, position_y_std, position_z_std 
            ) < self.config.position_std_threshold:
            return

        print('Start online reset...')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        print('+++publish:')
        print('   timestamp:', imu_state.timestamp)
        print('   orientation:', imu_state.orientation)
        print('   position:', imu_state.position)
        print('   velocity:', imu_state.velocity)
        print()
        
        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple('vio_result', ['timestamp', 'position', 'orientation', 'velocity', 'cam0_pose'])(
            time, imu_state.position, imu_state.orientation, body_velocity, T_c_w)