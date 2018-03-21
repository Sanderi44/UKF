import numpy as np
import math
import sys
from datetime import datetime
import matplotlib.pyplot as plt



def time_difference_in_seconds(d1, d2):
	d = d1-d2
	return float(d.seconds) + float(d.microseconds)/1000000.


def parse_sample_data(filename):
	t = []
	a = []
	g = []
	m = []
	q = []
	t0 = 0
	with open(filename, 'r') as f:
		c = 0
		for line in f:
			line = line.split("\t")[:-1]
			if len(line) > 1:
				c += 1
				if c > 2:
					datetime_object = datetime.strptime(line[0], '%Y/%m/%d %H:%M:%S.%f')
					if len(t) == 0:
						t0 = datetime_object
					t.append(time_difference_in_seconds(datetime_object, t0))
					a.append([float(line[1]), float(line[2]), float(line[3])])
					# print "A:", [float(line[1]), float(line[2]), float(line[3])]
					# print "G:", [float(line[4]), float(line[5]), float(line[6])]
					g.append([float(line[7]), float(line[8]), float(line[9])])
					m.append([float(line[10]), float(line[11]), float(line[12])])
					q.append([float(line[13]), float(line[14]), float(line[15]), float(line[16])])

	return np.array([t]).T, np.array(a), np.array(g), np.array(m), np.array(q)

def parse_sample_data_HIMU(filename):
	t = []
	a = []
	g = []
	m = []
	t0 = 0.
	with open(filename, 'r') as f:
		c = 0
		for line in f:
			line = line.strip().split(",")
			# print line
			# line = line.split("\t")[:-1]
			if len(line) > 1:
				c += 1
				if c > 2:
					t0 += 0.02

					t.append(t0)
					a.append([float(line[0]), float(line[1]), float(line[2])])
					# print "T:", t[-1]
					# print "A:", float(line[2]), float(line[3]), float(line[4])
					# print "G:", float(line[5]), float(line[6]), float(line[7])
					g.append([float(line[3]), float(line[4]), float(line[5])])
					m.append([float(line[6]), float(line[7]), float(line[8])])
					# 				q.append([float(line[13]), float(line[14]), float(line[15]), float(line[16])])

	return np.array([t]).T, np.array(a), np.array(g), np.array(m)

def parse_sample_data_OARS(filename):
	t = []
	a = []
	g = []
	m = []
	q = []
	t0 = 0
	lines = []
	with open(filename, 'r') as f:
		c = 0
		for line in f:
			line = line.split(",")[:-1]
			# print line
			if len(line) > 1:
				c += 1
				if c > 1:
					try:
						if line[3] != "" and line[4] != "" and line[5] != "" and line[6] != "":
							lines.append(line[1:8])
					except Exception as e:
						print e
						pass

	lines = sorted(lines, key=lambda x: float(x[0]))
	for line in lines:
		print line
		datetime_object = float(line[0])
		print datetime_object
		if len(t) == 0:
			t0 = datetime_object

		print datetime_object - t0

		t.append((datetime_object-t0)/1000.)
		a.append([float(line[1]), float(line[2]), float(line[3])])
		# print "T:", t[-1]
		# print "A:", float(line[2]), float(line[3]), float(line[4])
		# print "G:", float(line[5]), float(line[6]), float(line[7])
		g.append([float(line[4]), float(line[5]), float(line[6])])
		# 				m.append([float(line[10]), float(line[11]), float(line[12])])
		# 				q.append([float(line[13]), float(line[14]), float(line[15]), float(line[16])])



		# print c
	return np.array([t]).T, np.array(a), np.array(g)


def quat_product(r, q):
	res = [0.,0.,0.,0.]
	res[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	res[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
	res[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
	res[2] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
	return np.array(res)


def madgwick_prediction(Q, g, dt):
	gx = g[0]
	gy = g[1]
	gz = g[2]

	# print Q

	qw = Q[0]
	qx = Q[1]
	qy = Q[2]
	qz = Q[3]


	# qg = np.array([0., gx, gy, gz])
	# q_dot = 0.5*quat_product(Q, qg)
	q_dot = [0,0,0,0]
	q_dot[0] = 0.5 * (-qx * gx - qy * gy - qz * gz)
	q_dot[1] = 0.5 * (qw * gx + qy * gz - qz * gy)
	q_dot[2] = 0.5 * (qw * gy - qx * gz + qz * gx)
	q_dot[3] = 0.5 * (qw * gz + qx * gy - qy * gx)

	# print "before", Q
	Q_new = Q[:]
	Q_new[0] = Q[0] + q_dot[0]*dt
	Q_new[1] = Q[1] + q_dot[1]*dt 
	Q_new[2] = Q[2] + q_dot[2]*dt 
	Q_new[3] = Q[3] + q_dot[3]*dt
	# print "after", Q
	Q_new = normalize_quaterion(Q_new)
	return Q_new

def madgwick_update(Q, a, dt, beta):
	norm = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	ax = a[0]/norm
	ay = a[1]/norm
	az = a[2]/norm

	w = Q[0]
	x = Q[1]
	y = Q[2]
	z = Q[3]

	F = [[2*(x*z - w*y) - ax],
	  	 [2*(w*x + y*z) - ay],
		 [z*z - y*y - x*x* + w*w - az]]

	F = np.array(F)
	J = [[-2*y, 2*z, -2*w, 2*x],
		 [2*x, 2*w, 2*z, 2*y],
		 [2*w, -2*x, -2*y, 2*z]]
	J = np.array(J)
	J_T = J.T

	grad_F = np.matmul(J_T, F)

	s = grad_F/np.linalg.norm(grad_F)

	Q_new = Q[:]

	Q_new[0] -= beta * s[0,0] * dt;
	Q_new[1] -= beta * s[1,0] * dt;
	Q_new[2] -= beta * s[2,0] * dt;
	Q_new[3] -= beta * s[3,0] * dt;
	Q_new = normalize_quaterion(Q_new)
	return Q_new


def standard_integration(rpy, g, dt):
	rpy[0] = rpy[0] + g[0]*dt
	rpy[1] = rpy[1] + g[1]*dt
	rpy[2] = rpy[2] + g[2]*dt
	return rpy

def unscented_transformation(X, P, alpha=0.01, beta=2., kappa=0.1):
	# X is the state vector at time k-1, n x 1
	# P is the uncertainty matrix at time k-1, n x n
	# P[np.abs(P) < 10e-6] = 0.
	

	n = X.shape[0]

	lam = alpha*alpha*(n+kappa) - n

	all_sigma_points = []
	all_state_weights = []
	all_cov_weights = []
	# print P
	L = np.linalg.cholesky(P)
	# print "L", L
	X_sig_pos = np.zeros((L.shape[0], L.shape[1]))
	X_sig_neg = np.zeros((L.shape[0], L.shape[1]))
	W = lam/(n+lam)
	all_state_weights.append(W)
	W = (lam/(n+lam)) + (1.-alpha*alpha+beta)
	all_cov_weights.append(W)

	for i in range(0, L.shape[1]):
		# print (X + math.sqrt(n+lam)*np.array([L[:,i]]).T)
		# print (X - math.sqrt(n+lam)*np.array([L[:,i]]).T)
		X_sig_pos[:,i] = (X + math.sqrt(n+lam)*np.array([L[:,i]]).T)[:,0]
		X_sig_neg[:,i] = (X - math.sqrt(n+lam)*np.array([L[:,i]]).T)[:,0]
		W = 1./(2.*(n+lam))
		all_state_weights.append(W)
		all_state_weights.append(W)
		all_cov_weights.append(W)
		all_cov_weights.append(W)

	all_sigma_points = X
	all_sigma_points = np.hstack([all_sigma_points, X_sig_pos])
	all_sigma_points = np.hstack([all_sigma_points, X_sig_neg])
	# print all_sigma_points
	all_state_weights = np.array(all_state_weights)
	all_cov_weights = np.array(all_cov_weights)
	return all_sigma_points, all_state_weights, all_cov_weights


def calculate_F(g, dt):
	I = np.eye(4, dtype=float)
	# print g
	gx = g[0]
	gy = g[1]
	gz = g[2]

	F = I + 0.5*dt*np.array([[0.0, -gx, -gy, -gz], 
				   		     [gx,  0.0,  gz, -gy], 
				  		     [gy,  -gz,  0.0, gx], 
				  		     [gz,  gy, -gx,  0.0]])
	# F = np.array([[1.0, -gy*dt, -gx*dt, -gz*dt], [gy*dt, 1.0, gz*dt, -gx*dt], [gx*dt, -gz*dt, 1.0, gy*dt], [gz*dt, gx*dt, -gy*dt, 1.0]])
	return F


def prediction(F, sigma_points, state_weights, cov_weights, Q):
	n = sigma_points.shape[1]
	m = sigma_points.shape[0]
	state_mean = np.zeros((m, 1), dtype=float)
	state_cov = np.zeros((m, m), dtype=float)
	forward_project = np.zeros((m,n), dtype=float)
	for i in range(n):
		mat = np.matmul(F, np.array([sigma_points[:,i]]).T)
		forward_project[:,i] = mat[:,0]

	state_mean = np.matmul(forward_project, np.array([state_weights]).T)

	for i in range(n):
		W = cov_weights[i]
		# print np.array([sigma_points[:,i]]).T, state_mean
		X_diff = np.array([forward_project[:,i]]).T - state_mean
		X_diff_2 = np.matmul(X_diff, X_diff.T)

		cov = W*X_diff_2
		state_cov = state_cov + cov


	state_cov += Q
	return state_mean, state_cov





def update_RP(X, P, a, sigma_points, state_weights, cov_weights, R):
	norm = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	ax = a[0]/norm
	ay = a[1]/norm
	az = a[2]/norm


	n = sigma_points.shape[1]
	y_obs = np.array([[math.atan(ax/az)], [math.atan(ay/math.sqrt(ax*ax + az*az))]])
	y_i = np.zeros((2, n), dtype=float)
	y_mean = np.zeros((2, 1), dtype=float)


	for i in range(n):
		w = sigma_points[0,i]
		x = sigma_points[1,i]
		y = sigma_points[2,i]
		z = sigma_points[3,i] 
		roll_num = 2.*(w*x + y*z)
		roll_denom = 1. - 2.*(x*x + y*y)
		pitch_num = 2*(w*y - z*x)
		pitch_denom = w*w + x*x + y*y + z*z
		# h_x = np.array([[math.atan(roll_num/roll_denom)],
		# 				[math.asin(pitch_num/pitch_denom)]])

		y_i[0,i] = math.atan(roll_num/roll_denom)
		y_i[1,i] = math.asin(pitch_num/pitch_denom)
		y_mean += state_weights[i]*np.array([y_i[:,i]]).T
		

	Pyy = np.zeros((2, 2), dtype=float)
	Pxy = np.zeros((X.shape[0], 2), dtype=float)
	for i in range(n):
		y_diff = np.array([y_i[:,i]]).T - y_mean
		x_diff = np.array([sigma_points[:,i]]).T - X
		Pyy +=  cov_weights[i]*np.matmul(y_diff, y_diff.T)
		Pxy +=  cov_weights[i]*np.matmul(x_diff, y_diff.T)


	Pyy += R
	Pyy_inv = np.linalg.inv(Pyy)
	K = np.matmul(Pxy, Pyy_inv)
	y_diff_obs = y_obs - y_mean
	X = X + np.matmul(K, y_diff_obs)
	P = P - np.matmul(K, np.matmul(Pyy, K.T))

	return X, P




def EKF_prediction(F, X, P, Q):
	X = np.matmul(F, X)
	P = np.matmul(F, np.matmul(P, F.T)) + Q

	return X, P

innovations = []

def EKF_update_RP(X, P, a, R):
	w = X[0,0]
	x = X[1,0]
	y = X[2,0]
	z = X[3,0]

	norm = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	ax = a[0]/norm
	ay = a[1]/norm
	az = a[2]/norm

	w_2 = w*w
	x_2 = x*x
	y_2 = y*y
	z_2 = z*z

	roll_num = 2.*(w*x + y*z)
	roll_denom = 1. - 2.*(x*x + y*y)
	pitch_num = 2*(w*y - z*x)
	pitch_denom = w_2 + x_2 + y_2 + z_2

	h_x = np.array([[math.atan(roll_num/roll_denom)],
					[math.asin(pitch_num/pitch_denom)]])

	y_obs = np.array([[math.atan(ax/az)], [math.atan(ay/math.sqrt(ax*ax + az*az))]])

	# print h_x, y_obs


	H = np.zeros((2,4))




	denom_1 = 4*(w*x+y*z)*(w*x+y*z) + (-2*(x_2+y_2)+1)*(-2*(x_2+y_2)+1)

	H[0,0] = 2*x*(-2*(x_2+y_2)+1)/denom_1
	H[0,1] = 2*(w + 2*w*x_2 - 2*w*y_2 + 4*x*y*z)/denom_1
	H[0,2] = 2*(z - 2*x_2*z + 2*y_2*z + 4*w*x*y)/denom_1
	H[0,3] = 2*y*(-2*(x_2+y_2)+1)/denom_1


	u = pitch_num/pitch_denom
	mult = 1./math.sqrt(1-u*u)
	denom_2 = w_2 + x_2 + y_2 +z_2
	H[1,0] = mult * 2*(y_2*y - y*w_2 + y*x_2 + y*z_2 + 2*x*z*w)/denom_2
	H[1,1] = mult * 2*(-z_2*z - w_2*z + z*x_2 - y_2*z - 2*w*y*x)/denom_2
	H[1,2] = mult * 2*(w_2*w + w*x_2 - w*y_2 + w*z_2 + 2*x*z*y)/denom_2
	H[1,3] = mult * 2*(-x_2*x - w_2*x - y_2*x + x*z_2 - 2*w*y*z)/denom_2

	# print H
	# R = np.eye(R.shape[0])*0.001

	S = np.matmul(H, np.matmul(P, H.T)) + R
	S_inv = np.linalg.inv(S)
	K = np.matmul(P, np.matmul(H.T, S_inv))
	innovation = y_obs - h_x
	residual = innovation#np.array([euler[0:2, 0]]).T - y_obs

	X = X + np.matmul(K, innovation)
	I = np.eye(P.shape[0])
	P = np.matmul(I - np.matmul(K, H), P)
	euler = np.array([QuaterniontoEuler(X)]).T


	innovations.append(residual.tolist())

	return X, P


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


def QuaterniontoEuler(q):
	r, p, y = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])	
	return [r,p,y]

def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z


def normalize_quaterion(Q):
	norm = math.sqrt(Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])
	Q[0] = Q[0]/norm
	Q[1] = Q[1]/norm
	Q[2] = Q[2]/norm
	Q[3] = Q[3]/norm
	return Q


def main(args):
	shimmer = False
	# Parse data
	filename = args[1]
	if filename.startswith("OARS"):
		times, acc, gyro = parse_sample_data_OARS(filename)
	elif filename.startswith("HIMU"):
		times, acc, gyro, mag = parse_sample_data_HIMU(filename)
	elif filename.find("Shimmer") > -1:
		shimmer = True
		times, acc, gyro, mag, quat_true = parse_sample_data(filename)
	else:
		print "Invalid filename"
		return
	total_samples = times.shape[0]

	# Setup
	
	if shimmer:
		# Shimmer
		gyro_noise_std = np.deg2rad(0.0481)
		ax_std = 27.5e-3
		ay_std = 27.5e-3
		az_std = 27.5e-3

	else:
		# Pixel
		gyro_noise_std = np.deg2rad(0.07)
		ax_std = 1800*10e-6
		ay_std = 1800*10e-6
		az_std = 1800*10e-6

	# Accelerometer Characteristics
	ax_var = ax_std*ax_std
	ay_var = ay_std*ay_std
	az_var = az_std*az_std

	# Gyroscope Characteristics
	roll_var = gyro_noise_std*gyro_noise_std
	pitch_var = gyro_noise_std*gyro_noise_std
	yaw_var = gyro_noise_std*gyro_noise_std

	# Update Accelerometer Cutoff
	cutoff = 8.*ax_std
	print cutoff


	X = np.array([[1.],[0.],[0.],[0.]])
	P = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.0],[0., 0.0, 1., 0.0],[0., 0.0, 0.0, 1.]])
	Q_madg = [1., 0., 0., 0.]
	rpy = [0., 0., 0.]
	rpy_vals = []

	quat_madg = []
	quat_res = []
	quat_times = []

	prev_time = -(times[1,0] - times[0,0])

	a_prev = acc[0]
	update_times = []

	rpy_res = []
	rpy_true = []
	rpy_madg = []

	P_vals = []

	for i in range(total_samples):
		qw = X[0,0]
		qx = X[1,0]
		qy = X[2,0]
		qz = X[3,0]

		if filename.startswith("HIMU") != 1:
			gyros = np.deg2rad(gyro[i])
		else:
			gyros = gyro[i]

		gx = gyros[0]
		gy = gyros[1]
		gz = gyros[2]

		if shimmer:
			gyros[0] = gy
			gyros[1] = gz
			gyros[2] = gx


		# gyros = gyro[i]
		a = acc[i]
		norm = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
		if norm == 0.:
			continue
		ax = a[0]
		ay = a[1]
		az = a[2]

		a_diff = a - a_prev
		# print a_diff
		dt = times[i] - prev_time

		# rpy = standard_integration(rpy, gyros, dt[0])
		# rpy_vals.append(rpy)
		# if times[i] > 1.4:
		# 	break
		quat_times.append(times[i])

		# Create Q
		gyro_cov = np.array([[roll_var, 0.0, 0.0], [0.0, pitch_var, 0.0], [0.0, 0.0, yaw_var]])
		L = 0.5 * np.array([[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qz]])
		Q = np.matmul(L, np.matmul(gyro_cov, L.T))

		F = calculate_F(gyros, dt[0])
		
		# UKF Prediction
		sigma_points, state_weights, cov_weights = unscented_transformation(X, P)
		X, P = prediction(F, sigma_points, state_weights, cov_weights, Q)
		
		# EKF Prediction
		# X, P = EKF_prediction(F, X, P, Q)
		


		X = np.array([normalize_quaterion(X.T[0])]).T

		Q_madg = madgwick_prediction(Q_madg, gyros, dt[0])
		# Madgwick Update
		Q_madg = madgwick_update(Q_madg, a, dt[0], 0.2)
		quat_madg.append(Q_madg)
		rpy_madg_vals = QuaterniontoEuler(Q_madg)
		rpy_madg.append(rpy_madg_vals)
		# print X
		# UPDATE



		if (abs(a_diff[0]) < cutoff and abs(a_diff[1]) < cutoff and abs(a_diff[2]) < cutoff):
			update_times.append(times[i,0])

			# Calculate R
			M = np.array([[az/(ax*ax + az*az), 0.0, -ax/(ax*ax + az*az)], [-ay*ax/(ax*ax + ay*ay + az*az)*math.sqrt(ax*ax + az*az), math.sqrt(ax*ax + az*az)/(ax*ax + ay*ay + az*az), -ay*az/((ax*ax + ay*ay + az*az)*math.sqrt(ax*ax + az*az))]])
			R_l = np.array([[ax_var, 0.0, 0.0], [0.0, ay_var, 0.0], [0.0, 0.0, az_var]])
			R = np.matmul(M, np.matmul(R_l, M.T))
			
			# EKF Update
			# X, P = EKF_update_RP(X, P, a, R)



			# # UKF Update
			sigma_points, state_weights, cov_weights = unscented_transformation(X, P)
			X, P = update_RP(X, P, a, sigma_points, state_weights, cov_weights, R)

			X = np.array([normalize_quaterion(X.T[0])]).T


		res = [X[0, 0], X[1, 0], X[2, 0], X[3, 0]]
		quat_res.append(res)
		prev_time = times[i]
		a_prev = a
		res_rpy = QuaterniontoEuler(res)
		rpy_res.append(res_rpy)
		P_vals.append([P[0, 0], P[1, 1], P[2, 2], P[3, 3]])

		try:
			res_true = QuaterniontoEuler(quat_true[i,:])
			rpy_true.append(res_true)
		except:
			pass

	plt.figure()
	plt.plot(quat_times, np.array(quat_res)[:,0], label="q0")
	plt.plot(quat_times, np.array(quat_res)[:,1], label="q1")
	plt.plot(quat_times, np.array(quat_res)[:,2], label="q2")
	plt.plot(quat_times, np.array(quat_res)[:,3], label="q3")
	plt.title("Estimated Q")
	plt.legend()


	# print np.array(quat_madg)[:,0]
	plt.figure()
	plt.plot(quat_times, np.array(quat_madg)[:,0], label="q0")
	plt.plot(quat_times, np.array(quat_madg)[:,1], label="q1")
	plt.plot(quat_times, np.array(quat_madg)[:,2], label="q2")
	plt.plot(quat_times, np.array(quat_madg)[:,3], label="q3")
	plt.title("Madgwick Q")
	plt.legend()


	try:
		plt.figure()
		plt.plot(quat_times, np.array(quat_true)[:len(quat_times), 0], label="q0")
		plt.plot(quat_times, np.array(quat_true)[:len(quat_times), 1], label="q1")
		plt.plot(quat_times, np.array(quat_true)[:len(quat_times), 2], label="q2")
		plt.plot(quat_times, np.array(quat_true)[:len(quat_times), 3], label="q3")
		plt.title("True Q")
		plt.legend()
	except:
		plt.close()
		pass

	# plt.figure()
	# plt.plot(quat_times, np.array(rpy_vals)[:len(quat_times), 0], label="r")
	# plt.plot(quat_times, np.array(rpy_vals)[:len(quat_times), 1], label="p")
	# plt.plot(quat_times, np.array(rpy_vals)[:len(quat_times), 2], label="y")
	# plt.title("RPY")
	# plt.legend()

	try:
		plt.figure()
		plt.plot(quat_times, np.array(gyro)[:, 0], label="r")
		plt.plot(quat_times, np.array(gyro)[:, 1], label="p")
		plt.plot(quat_times, np.array(gyro)[:, 2], label="y")
		plt.title("Gyros")
		plt.legend()
	except:
		plt.close()

	try:
		plt.figure()
		plt.plot(quat_times, np.array(acc)[:, 0], label="x")
		plt.plot(quat_times, np.array(acc)[:, 1], label="y")
		plt.plot(quat_times, np.array(acc)[:, 2], label="z")
		plt.title("Acc")
		plt.legend()
	except:
		plt.close()
	# print np.array(innovations)
	# plt.figure()
	# plt.plot(np.array(innovations)[:,0], '.', label="Col 1")
	# plt.plot(np.array(innovations)[:,1], '.', label="Col 2")
	# plt.legend()
	# plt.title("Residual")


	plt.figure()
	plt.plot(quat_times, np.array(rpy_res)[:,0], label="Roll")
	plt.plot(quat_times, np.array(rpy_res)[:,1], label="Pitch")
	plt.plot(quat_times, np.array(rpy_res)[:,2], label="Yaw")
	plt.legend()
	plt.title("Estimated RPY")

	plt.figure()
	plt.plot(quat_times, np.array(rpy_madg)[:,0], label="Roll")
	plt.plot(quat_times, np.array(rpy_madg)[:,1], label="Pitch")
	plt.plot(quat_times, np.array(rpy_madg)[:,2], label="Yaw")
	plt.legend()
	plt.title("Madgwick RPY")

	try:
		plt.figure()
		plt.plot(quat_times, np.array(rpy_true)[:,0], label="Roll")
		plt.plot(quat_times, np.array(rpy_true)[:,1], label="Pitch")
		plt.plot(quat_times, np.array(rpy_true)[:,2], label="Yaw")
		plt.legend()
		plt.title("True RPY")
	except:
		plt.close()
		pass


	plt.figure()
	plt.plot(quat_times, np.array(P_vals)[:,0], label="q0")
	plt.plot(quat_times, np.array(P_vals)[:,1], label="q1")
	plt.plot(quat_times, np.array(P_vals)[:,2], label="q2")
	plt.plot(quat_times, np.array(P_vals)[:,2], label="q3")
	plt.legend()
	plt.title("Estimated Variance")

	plt.show()






if __name__ == "__main__":
	main(sys.argv)