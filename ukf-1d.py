import numpy as np
import math

def unscented_transformation(X, P, lam=0.1, alpha=1., beta=2.):
	# X is the state vector at time k-1, n x 1
	# P is the uncertainty matrix at time k-1, n x n
	n = X.shape[0]

	all_sigma_points = []
	all_state_weights = []
	all_cov_weights = []

	for i in range(len(X[:, 0])):
		state = X[i, 0]
		
		sigma_points = []
		state_weights = []
		cov_weights = []
		

		W = lam/(n+lam)
		sigma_points.append(state)
		state_weights.append(W)
		cov_weights.append(W + (1. - alpha*alpha + beta))
		
		W = lam/(2*(n+lam))
		for j in range(0, n):
			spread = (n+lam)*P[i, j]
			if spread < 0.:
				spread = 0.
			state_sigma = state + np.sqrt(spread)
			sigma_points.append(state_sigma)
			state_weights.append(W)
			cov_weights.append(W)
		for j in range(0, n):
			spread = (n+lam)*P[i, j]
			if spread < 0.:
				spread = 0.
			state_sigma = state - np.sqrt(spread)
			sigma_points.append(state_sigma)	
			state_weights.append(W)
			cov_weights.append(W)
		all_sigma_points.append(sigma_points)
		all_state_weights.append(state_weights)
		all_cov_weights.append(cov_weights)
	all_sigma_points = np.array(all_sigma_points)
	all_state_weights = np.array(all_state_weights)
	all_cov_weights = np.array(all_cov_weights)
	return all_sigma_points, all_state_weights, all_cov_weights


def calculate_F(g, dt):
	I = np.eye(4, dtype=float)
	gx = g[0]
	gy = g[1]
	gz = g[2]

	A = np.array([[0.0, -gx, -gy, -gz], [gx, 0.0, gz, -gy], [gy, gz, 0.0, gx], [gz, gy, gx, 0.0]])
	return I+A*dt


def prediction(F, sigma_points, state_weights, cov_weights, Q):
	n = sigma_points.shape[1]
	m = sigma_points.shape[0]
	state_mean = np.zeros((m, 1), dtype=float)
	state_cov = np.zeros((m, m))

	for i in range(n):
		mat = np.matmul(F, np.array([sigma_points[:,i]]).T)
		W = np.array([state_weights[:,i]]).T
		state_mean += mat*W

	for i in range(n):
		W = np.array([cov_weights[:,i]]).T
		X_diff = np.array([sigma_points[:,i]]).T - state_mean
		W_X = W*X_diff
		cov = np.matmul(W_X, X_diff.T)
		state_cov += cov


	return state_mean, state_cov + Q


def RPYtoQuaternion(roll, pitch, yaw):
	cy = cos(yaw * 0.5);
	sy = sin(yaw * 0.5);
	cr = cos(roll * 0.5);
	sr = sin(roll * 0.5);
	cp = cos(pitch * 0.5);
	sp = sin(pitch * 0.5);

	q[0] = cy * cr * cp + sy * sr * sp
	q[1] = cy * sr * cp - sy * cr * sp
	q[2] = cy * cr * sp + sy * sr * cp
	q[3] = sy * cr * cp - cy * sr * sp
	return np.array([q]).T





def update_RP(X, P, a, sigma_points, state_weights, cov_weights, R):
	n = sigma_points.shape[1]
	y = np.array([[0.], [0.]])
	y_obs = np.array([[math.atan2(a[1], a[2])], [math.atan2(-a[0], np.sqrt(a[1]*a[1] + a[2]*a[2]))]])
	y_i = []

	for i in range(n):
		x = sigma_points[:,i]
		h_x = np.array([[math.atan2(2*(x[0]*x[1] + x[2]*x[3]), (x[0]*x[0] - x[1]*x[1] - x[2]*x[2] - x[3]*x[3]))],
			   [math.asin(2*(x[0]*x[3] - x[1]*x[2])/(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]))]])
		y_i.append(h_x)
		y += h_x/n

	Pyy = np.zeros((2, 2), dtype=float)
	Pxy = np.zeros((4, 2), dtype=float)
	for i in range(n):
		y_diff = y_i[i] - y
		x_diff = np.array([sigma_points[:,i]], dtype=float).T - X
		x_diff_w = (1./2.*n)*x_diff
		Pyy +=  np.matmul((1./2.*n)*y_diff, y_diff.T)
		pxy = np.matmul(x_diff, y_diff.T)
		Pxy +=  pxy

	Pyy += R

	Pyy_inv = np.linalg.inv(Pyy)
	K = np.matmul(Pxy, Pyy_inv)
	
	y_diff_obs = y_obs - y
	X = X + np.matmul(K, y_diff_obs)
	P = P - np.matmul(K, np.matmul(Pyy, K.T))
	return X, P


def main():
	# Setup
	X = np.array([[1.],[0.],[0.],[0.]])
	P = np.array([[0.1, 0., 0., 0.],[0., 0.1, 0., 0.0],[0., 0.0, 0.1, 0.0],[0., 0.0, 0.0, 0.1]])
	# print X
	sigma_points, state_weights, cov_weights = unscented_transformation(X, P)

	qw = X[0,0]
	qx = X[1,0]
	qy = X[2,0]
	qz = X[3,0]
	roll_var = 0.07*0.07
	pitch_var = 0.07*0.07
	yaw_var = 0.07*0.07
	gyros = [0.01, 0.02, 0.03]
	gyro_cov = np.array([[roll_var, 0.0, 0.0], [0.0, pitch_var, 0.0], [0.0, 0.0, yaw_var]])
	L = 0.5 * np.array([[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qz]])
	Q = np.matmul(L, np.matmul(gyro_cov, L.T))

	# print Q


	# Q = np.array([[0.07, 0.0, 0.0, 0.0], [0.0, 0.07, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0], [0.0, 0.0, 0.0, 0.07]])

	dt = 0.01
	F = calculate_F(gyros, dt)

	X, P = prediction(F, sigma_points, state_weights, cov_weights, Q)
	# print X



	sigma_points, state_weights, cov_weights = unscented_transformation(X, P)

	ax = 0.9
	ay = 0.1
	az = -0.1
	a = np.array([ax, ay, az])
	ax_std = 0.011718
	ay_std = 0.011718
	az_std = 0.0171875

	ax_var = ax_std*ax_std
	ay_var = ay_std*ay_std
	az_var = az_std*az_std


	M = np.array([[0.0, az/(az*az), -az/(az*az)], [-1/(np.sqrt(ay*ay + az*az)), ax*ay/((ax*ax + ay*ay + az*az)*np.sqrt(ax*ax + az*az)), ax*az/((ax*ax + ay*ay + az*az)*np.sqrt(ax*ax + ay*ay))]])
	R_l = np.array([[ax_var, 0.0, 0.0], [0.0, ay_var, 0.0], [0.0, 0.0, az_var]])

	R = np.matmul(M, np.matmul(R_l, M.T))

	X, P = update_RP(X, P, a, sigma_points, state_weights, cov_weights, R)


if __name__ == "__main__":
	main()