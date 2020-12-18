import numpy as np
from scipy.linalg import expm, block_diag
from utils import *

# Returns the hat of given vector
def get_hat(x):

	return np.array([[0,-x[2],x[1]],
	[x[2],0,-x[0]],
	[-x[1],x[0],0]])

# This  function takes linear and rotational velocities as input and returns
# u_hat and u_curlyhat as the outputs (uses get_hat() function)
def get_operators(v,w):

	w_hat = get_hat(w)
	v_hat = get_hat(v)

	u_curlyhat = np.block([[w_hat,v_hat],
	[np.zeros((3,3)),w_hat]])

	u_hat = np.block([[w_hat,v.reshape((3,1))],
	[np.zeros((1,3)),0]])

	return u_hat,u_curlyhat

# Used to get the corresponding flat indices of 2D array
def get_flat_indices(true_columns,size_M):

	boolean_array = np.full((3,size_M),False)
	boolean_array[:,true_columns] = True
	return np.where( (boolean_array.T).reshape((1,3*size_M)) )[1]

# Used to get the dot operator of s (dot in circle)
def get_dot_operator(s):

	max_size = s.shape[1]
	s_dot = np.zeros((4,6,max_size))
	s_dot[0,0,:] = 1
	s_dot[1,1,:] = 1
	s_dot[2,2,:] = 1
	s_dot[0,4,:] = s[2,:]
	s_dot[0,5,:] = -s[1,:]
	s_dot[1,3,:] = -s[2,:]
	s_dot[1,5,:] = s[0,:]
	s_dot[2,3,:] = s[1,:]
	s_dot[2,4,:] = -s[0,:]

	return s_dot

# Transform from 6x1 representation to 4x4 
# (SE3 and twistmatrix lie algebra)
def get_lie_algebra(xi):

	xi_hat = np.zeros((4,4))
	xi_hat[:3,:3] = get_hat(xi[3:,0])
	xi_hat[:3,3] = xi[:3,0]

	return xi_hat

# Get the best_max features from M (total) features
def get_best_features(features,best_max,t):

	best_features = np.zeros( (features.shape[0],best_max,features.shape[2]) )
	feature_count = np.zeros( features.shape[1] )
	# Checks number of time each features is observed
	for t_step in range(1,np.size(t)):
		indGood = np.sum(features[:,:,t_step],axis=0)!=-4
		feature_count[indGood] = feature_count[indGood] + 1

	# Get the best_max number of features observed 
	# That is, if a feature is observed more number of time it is more likely to be included for EKF
	ind_best_features = np.sort( np.flip( np.argsort(feature_count) )[:best_max] )
	best_features = features[:,ind_best_features,:]
	return best_features




if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# Get 1000 best features (found to work well in most cases)
	features = get_best_features(features,1000,t)
	# Build the calibration matrix M (stereo) from K matrix
	block_cam = K[:2]
	M_matrix = np.block([
		[block_cam,np.zeros((2,1))],
		[block_cam,np.array([[-K[0,0]*b],[0]])]
	])

	# Transformation from regular frame to optical frame
	oRr = np.array([
		[0,-1,0],
		[0,0,-1],
		[1,0,0]
	])
	oRr_inv = np.linalg.inv(oRr)

	# Inverse of cam_T_imu
	cam_T_imu_inv = np.linalg.inv(cam_T_imu)

	#-----------------------------------------------------------------------
	
	## Initialize parameters for part (a)
	mu_tt = np.eye(4)
	sigma_tt = np.eye(6)*1
	# motion_noise = np.diag([0.5,0.5,0.5,0.05,0.05,0.05])*3
	motion_noise = np.eye(6)*0.001

	# Store both the world_T_imu and it's inverse for later use
	world_T_imu = np.zeros((4,4,np.size(t)))
	world_T_imu[:,:,0] = np.eye(4)
	world_T_imu_inv = np.zeros((4,4,np.size(t)))
	world_T_imu_inv[:,:,0] = np.eye(4)

	## Initialize parameters for part (b)	
	# Get the total number of features
	M_max = np.shape(features)[1]
	# Flat matrix for landmark positions
	m_j_flat = np.full( (1,3*M_max),np.NaN )
	# 3xM matrix for landmark positions
	m_j = m_j_flat.reshape((M_max,3)).T
	# Projection matrix
	projection_matrix = np.block( [np.eye(3),np.zeros((3,1))] )
	# Mu and Sigma for part (b)
	sigma_map_t = np.eye(3*M_max)*100
	# Both flattened and 3xM verisons are maintained for vectorizing the code
	mu_map_t = np.zeros((3*M_max,1))
	mu_map_2d = np.zeros((3,M_max))
	# Observation Noise
	v_noise = 15

	for t_step in range(1,np.size(t)):

		### Part(a) IMU Localization via EKF Prediction
		# Get tau
		tau = t[0][t_step] - t[0][t_step-1]
		# Get the linear velocity from data at this timestep
		v_t = linear_velocity[:,t_step]
		# Get the rotational velocity from data at this timestep
		w_t = rotational_velocity[:,t_step]

		# Get u_hat and u_curlyhat using function
		u_hat,u_curlyhat = get_operators(v_t,w_t)

		# Apply EKF equations to predict inverse IMU pose and covariance
		mu_predict = np.matmul(expm((-tau*u_hat)),mu_tt)
		rot_curly = expm((-tau*u_curlyhat))
		sigma_predict = np.matmul(rot_curly,sigma_tt)
		sigma_predict = np.matmul(sigma_predict,rot_curly.T) + motion_noise
		# Store the IMU pose for use in part (b)
		world_T_imu_predict = np.linalg.inv(mu_predict)

		### Part(b) Landmark Mapping via EKF Update
		# Get the observed features at this timestep
		indGood = np.sum(features[:,:,t_step],axis=0)!=-4
		# Get all the features which have not been observed yet
		new_mj = np.isnan( np.sum(m_j,axis=0) )
		# Get all the features which have been observed previously
		old_mj = np.invert(new_mj)
		# Get the features which have not been observed yet at this timestep
		new_good_mj = np.where(np.logical_and(indGood,new_mj))[0]
		# Get the features which have been observed previously at this timestep
		ind_old_good_mj = np.where(np.logical_and(indGood,old_mj))[0]

		# If there are new features at this timestep, prior needs to be initialized
		if new_good_mj.size!=0:

			# Get the pixel coordinates	
			u_L = features[0,new_good_mj,t_step]
			v_L = features[1,new_good_mj,t_step]
			u_R = features[2,new_good_mj,t_step]

			# Get (x,y,z) in optical frame using Stereo Camera Model
			x_m = (u_L - M_matrix[0,2])/M_matrix[0,0]
			y_m = (v_L - M_matrix[1,2])/M_matrix[1,1]
			z_m = M_matrix[2,3]/(u_R-u_L)
			xyz_o = np.multiply(z_m,np.block([[x_m],[y_m],[np.ones(x_m.size)],[1/z_m]]))

			# Update the landmark positions (basically set prior)
			# m_j only has prior values, not updated anywhere else, 
			# mu_map which also stores the necessary values of prior is updated later
			m_j[:,new_good_mj] = (world_T_imu_predict @ cam_T_imu_inv @ xyz_o)[:3,:]
			# Also update the flattened array (useful later in part (b))
			new_good_mj_flat = get_flat_indices(new_good_mj,M_max)
			mu_map_t[new_good_mj_flat,:] =  (m_j.T).reshape((3*M_max,1))[new_good_mj_flat,:]
			mu_map_2d = mu_map_t.reshape(3,M_max,order='F')
		
		# If there are features observed which were previously obesrved, we'll update them
		if ind_old_good_mj.size!=0:
			
			# Get flat indices
			old_good_mj_flat = get_flat_indices(ind_old_good_mj,M_max)
			# Nt is the number of features observed at this timestep which were already observed
			Nt = np.size(ind_old_good_mj)
			# Get the pixel coordinates and flatten it
			z_t = (features[:,ind_old_good_mj,t_step]).reshape(-1,1,order = 'F')
			# Homogenize landmark positions
			old_good_mj = np.vstack( ( mu_map_2d[:,ind_old_good_mj],np.ones(Nt) ) )
			# Get oTixwTixmj
			q_z = cam_T_imu @ mu_predict @ old_good_mj
			# Get z_tilda (predicted observations)
			z_tilda_t = (M_matrix @ q_z/q_z[2,:]).reshape(-1,1,order = 'F')
			# Compute the Jacobian and get Ht matrix
			delpi_delq = np.zeros((4,4,Nt))
			delpi_delq[0,0,:] = 1/q_z[2,:]
			delpi_delq[1,1,:] = 1/q_z[2,:]
			delpi_delq[3,3,:] = 1/q_z[2,:]
			delpi_delq[0,2,:] = -q_z[0,:]/np.square(q_z[2,:])
			delpi_delq[1,2,:] = -q_z[1,:]/np.square(q_z[2,:])
			delpi_delq[3,2,:] = -1/np.square(q_z[2,:])
			post_matrix = cam_T_imu @ mu_predict @ projection_matrix.T
			temp_matrix = np.einsum('ijk,jl->ilk',delpi_delq,post_matrix)
			# Using 3D matrix just for vectorizing code. Is later converted to 2D
			Ht_3dmatrix = np.einsum('ij,jkl->ikl',M_matrix,temp_matrix)
			Ht_2dmatrix = block_diag(*Ht_3dmatrix.T).T

			# Apply EKF update equations for mu and sigma of landmarks
			mu_map_t_good = mu_map_t[old_good_mj_flat,:]
			row_indices = np.tile(old_good_mj_flat,(3*Nt,1)).reshape(-1,1,order='F')
			column_indices = np.tile(old_good_mj_flat,(3*Nt,1)).reshape(-1,1)
			sigma_map_t_good = sigma_map_t[row_indices,column_indices].reshape(3*Nt,3*Nt)
			# S matrix
			S_matrix = (Ht_2dmatrix @ sigma_map_t_good @ Ht_2dmatrix.T) + np.eye(4*Nt)*v_noise
			# Kalman Gain
			Kt = sigma_map_t_good @ Ht_2dmatrix.T @ np.linalg.inv(S_matrix)
			# Update mu
			mu_map_t1_good = mu_map_t_good + ( Kt @ (z_t - z_tilda_t) )
			# Update sigma
			sigma_map_t1_good = (np.eye(3*Nt) - (Kt @ Ht_2dmatrix) ) @ sigma_map_t_good
			# Since mu in update step was 3Mx1, 3xM is required for some computations 
			# Hence, maintianing 2D array as well
			mu_map_t[old_good_mj_flat,:] = mu_map_t1_good
			mu_map_2d = mu_map_t.reshape(3,M_max,order='F')
			sigma_map_t[row_indices,column_indices] = sigma_map_t1_good.reshape(-1,1)

			### Part(c) Visual-Inertial SLAM
			
			# Get the dot operation (dot inside a circle)
			mu_map_imu = mu_predict @ old_good_mj
			mu_map_imu_dot = get_dot_operator(mu_map_imu)
			# Get the Ht matrix for part (c)
			temp2_matrix = np.einsum('ij,jkl->ikl',cam_T_imu,mu_map_imu_dot)
			temp3_matrix = np.einsum('ijk,jlk->ilk',delpi_delq,temp2_matrix)
			H_imu = np.einsum('ij,jkl->ikl',M_matrix,temp3_matrix)
			H_imu_2d = np.transpose(H_imu,(2,0,1)).reshape(4*Nt,6)

			# Apply EKF equations
			# S matrix
			S_imu_matrix = H_imu_2d @ sigma_predict @ H_imu_2d.T + np.eye(4*Nt)*v_noise
			# Kalman Gain
			K_imu = sigma_predict @ H_imu_2d.T @ np.linalg.inv(S_imu_matrix)
			# xi is 6x1 which is to be converted to 4x4 (SO3) using lie algebra
			xi = K_imu @ (z_t - z_tilda_t)
			xi_hat = get_lie_algebra(xi)
			# Update mu
			mu_tt = expm(xi_hat) @ mu_predict
			# Update sigma
			sigma_tt = (np.eye(6) - (K_imu @ H_imu_2d) ) @ sigma_predict

			## Update the landmark positions using updatetd IMU position
			# Get oTixwTixmj
			q_z = cam_T_imu @ mu_tt @ old_good_mj
			# Get z_tilda (predicted observations)
			z_tilda_t = (M_matrix @ q_z/q_z[2,:]).reshape(-1,1,order = 'F')
			# Compute the Jacobian and get Ht matrix
			delpi_delq = np.zeros((4,4,Nt))
			delpi_delq[0,0,:] = 1/q_z[2,:]
			delpi_delq[1,1,:] = 1/q_z[2,:]
			delpi_delq[3,3,:] = 1/q_z[2,:]
			delpi_delq[0,2,:] = -q_z[0,:]/np.square(q_z[2,:])
			delpi_delq[1,2,:] = -q_z[1,:]/np.square(q_z[2,:])
			delpi_delq[3,2,:] = -1/np.square(q_z[2,:])
			post_matrix = cam_T_imu @ mu_tt @ projection_matrix.T
			temp_matrix = np.einsum('ijk,jl->ilk',delpi_delq,post_matrix)
			# Using 3D matrix just for vectorizing code. Is later converted to 2D
			Ht_3dmatrix = np.einsum('ij,jkl->ikl',M_matrix,temp_matrix)
			Ht_2dmatrix = block_diag(*Ht_3dmatrix.T).T

			# Apply EKF update equations for mu and sigma of landmarks
			mu_map_t_good = mu_map_t[old_good_mj_flat,:]
			row_indices = np.tile(old_good_mj_flat,(3*Nt,1)).reshape(-1,1,order='F')
			column_indices = np.tile(old_good_mj_flat,(3*Nt,1)).reshape(-1,1)
			sigma_map_t_good = sigma_map_t[row_indices,column_indices].reshape(3*Nt,3*Nt)
			# S matrix
			S_matrix = (Ht_2dmatrix @ sigma_map_t_good @ Ht_2dmatrix.T) + np.eye(4*Nt)*v_noise
			# Kalman Gain
			Kt = sigma_map_t_good @ Ht_2dmatrix.T @ np.linalg.inv(S_matrix)
			# Update mu
			mu_map_t1_good = mu_map_t_good + ( Kt @ (z_t - z_tilda_t) )
			# Update sigma
			sigma_map_t1_good = (np.eye(3*Nt) - (Kt @ Ht_2dmatrix) ) @ sigma_map_t_good
			# Since mu in update step was 3Mx1, 3xM is required for some computations 
			# Hence, maintianing 2D array as well
			mu_map_t[old_good_mj_flat,:] = mu_map_t1_good
			mu_map_2d = mu_map_t.reshape(3,M_max,order='F')
			sigma_map_t[row_indices,column_indices] = sigma_map_t1_good.reshape(-1,1)

		# If no previously overved features are observed at this time step, 
		# we use our predicted mu and sigma for prediction in next time step
		# In reality it is not always Predict,Update,Predict Update... It can be Predict Predict Predict Update....
		# This condition takes such things into account	
		else:

			mu_tt = mu_predict
			sigma_tt = sigma_predict
		# Store the IMU pose for plot
		world_T_imu[:,:,t_step] = np.linalg.inv(mu_tt)

	# Visualize the trajectory
	visualize_trajectory_2d(world_T_imu,mu_map_2d,show_ori=False)


