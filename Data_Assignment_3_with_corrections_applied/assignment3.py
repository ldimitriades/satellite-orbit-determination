import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

t = np.loadtxt('t.txt') # epochs - s
tim = t - t[0]

CA_range = np.loadtxt('CA_range.txt') # pseudorange observations from CA code - km

PRN_ID = np.loadtxt('PRN_ID.txt') # PRN ID of tracked GPS satellites

rx_gps = np.loadtxt('rx_gps.txt') # GPS satellite positions (transmitters) - km
ry_gps = np.loadtxt('ry_gps.txt')
rz_gps = np.loadtxt('rz_gps.txt')



vx_gps = np.loadtxt('vx_gps.txt') # GPS satellite velocities (transmitters) - km/s
vy_gps = np.loadtxt('vy_gps.txt')
vz_gps = np.loadtxt('vz_gps.txt')

rx = np.loadtxt('rx.txt') # precise positions (receivers) - km
ry = np.loadtxt('ry.txt')
rz = np.loadtxt('rz.txt')

vx = np.loadtxt('vx.txt') # precise velocities (receivers) - km/s
vy = np.loadtxt('vy.txt')
vz = np.loadtxt('vz.txt')


def obs_cov_matrix(size):
    sigma = 0.003
    correlation = 0
    cov_matrix = np.zeros((size, size))
    for l in range(size):
        for m in range(size):
            if l == m:
                cov_matrix[l, m] = sigma ** 2
            else:
                cov_matrix[l, m] = correlation * (sigma ** 2)

    return cov_matrix

def state_cov_matrix(sigma_pos, sigma_vel, correlation_pos_vel):
	cov_matrix = np.zeros((6,6))
	for l in range(6):
		for m in range(6):
			if l <= 2 and m <=2:
				if l == m:
					cov_matrix[l,m] = sigma_pos**2
			else:
				if l==m:
					cov_matrix[l,m] = sigma_vel**2
				if abs(l-m) == 3:
					cov_matrix[l,m] = sigma_pos*sigma_vel*correlation_pos_vel
	return cov_matrix

def least_squares(x_0, rx_gps, ry_gps, rz_gps, CA_range):
	i=0
	obs = np.trim_zeros(CA_range)
	size = len(obs)
	x_r = x_0[0]
	y_r = x_0[1]
	z_r = x_0[2]
	rx_gps = np.trim_zeros(rx_gps)
	ry_gps = np.trim_zeros(ry_gps)
	rz_gps = np.trim_zeros(rz_gps)
	f_x_matrix = np.zeros((1, size))
	par_x_matrix = np.zeros((1, size))
	par_y_matrix = np.zeros((1, size))
	par_z_matrix = np.zeros((1, size))
	par_vx_matrix = np.zeros((1, size))
	par_vx_matrix = par_vx_matrix.reshape(size,1)
	par_vy_matrix = np.zeros((1, size))
	par_vy_matrix = par_vy_matrix.reshape(size,1)
	par_vz_matrix = np.zeros((1, size))
	par_vz_matrix = par_vz_matrix.reshape(size,1)
	while(i<size):

			f_x = ((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5 
			part_rx_receiver = (x_r - rx_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			part_ry_receiver = (y_r - ry_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			part_rz_receiver = (z_r - rz_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			f_x_matrix[0,i] = f_x.item()
			par_x_matrix[0,i] = part_rx_receiver.item()
			par_y_matrix[0,i] = part_ry_receiver.item()
			par_z_matrix[0,i] = part_rz_receiver.item()
			i+=1

	par_x_matrix = par_x_matrix.reshape(size,1)
	par_y_matrix = par_y_matrix.reshape(size,1)
	par_z_matrix = par_z_matrix.reshape(size,1)
	H_matrix = np.hstack([par_x_matrix,par_y_matrix,par_z_matrix, par_vx_matrix,par_vy_matrix,par_vz_matrix])
	delta_z = obs - f_x_matrix
	delta_z = delta_z.reshape(size,1)
	
	return H_matrix, delta_z

def updatey(obs_cov_matrix,state_cov_matrix, y,i):
	size = len(np.trim_zeros(CA_range[i]))
	obs_cov_matrix = obs_cov_matrix(size)
	y_0 = y
	ep_mat = np.zeros((12,1))
	H, delta_z = least_squares(y_0, rx_gps[i], ry_gps[i], rz_gps[i], CA_range[i])
	H_t = np.transpose(H)
	product1 = np.dot(H, np.dot(state_cov_matrix, H_t))
	term2 = product1 + obs_cov_matrix
	term2_inv = np.linalg.inv(term2)
	term3 = np.dot(state_cov_matrix, H_t)
	K = np.dot(term3,term2_inv)
	term4 = np.dot(K, delta_z)
	y_0 += term4
	term5 = (np.eye(6) - np.dot(K,H))
	term6 = np.dot(K,delta_z)
	state_cov_matrix = np.dot(term5,state_cov_matrix)
	ep = delta_z - np.dot(H,term6)
	ep_r = ep.flatten()**2
	ep_rms = (np.sum(ep_r)/size)**0.5

	return state_cov_matrix, y_0, ep_rms

def dydt(t,y):
	
	dydt = np.zeros_like(y)
	Phi = y[6:]
	Phi = Phi.reshape((6, 6))
	r = y[0:3]
	v = y[3:6]
	
	#constants
	C_D = 2.6  
	omega = 7.292115e-5  #rad/s
	rho = 1e-2  #kg/km^3
	A = 1e-6  #km^2
	GM = 3.986004415e5   #km^3/s^2
	m = 500  #kg
	drag_const = 0.5*A*C_D*rho/m
	Omega_matrix = np.array([[0, omega, 0],[-omega, 0, 0],[0, 0, 0]])
	r_mag = np.linalg.norm(r)
	
	v_mag = np.linalg.norm(v)
	drag = -drag_const*(v_mag)
	grav = -GM / (r_mag**3)
	a_centr = - np.dot(Omega_matrix, np.dot(Omega_matrix,r))
	a_cori = 2*np.dot(Omega_matrix, v)
	a = a_centr + a_cori + grav*r #+ drag*v
	diff_state = np.hstack([v,a])

	par_a_r = - np.dot(Omega_matrix,Omega_matrix)*np.eye(3) - (GM / r_mag**3) * np.eye(3) + (3 * GM / r_mag**5) * np.outer(r, r)
	par_a_v = 2 * Omega_matrix * np.eye(3) #-drag_const * v_mag * np.eye(3) + (drag_const / v_mag) * np.outer(v, v)
	df_dy = np.zeros((6,6))
	df_dy[:3,3:] = np.eye(3)
	df_dy[3:,:3] = par_a_r
	df_dy[3:,3:] = par_a_v
	dPhi_dt = np.dot(df_dy,Phi)
	
	
	dydt = np.hstack([diff_state, dPhi_dt.flatten()])
	dydt = dydt

	return dydt

def kalman_filter(precise_state, obs_cov_matrix, state_cov_matrix, Q_k):
	
	state_cov_matrix1 = state_cov_matrix(0.002,0.0001,.7)
	state_cov_matrix1,y_0,ep = updatey(obs_cov_matrix ,state_cov_matrix1, precise_state, 0)
	initial_state = np.zeros((42,1))
	initial_state[:6] = y_0
	diagonal_initial = np.diag([1,1,1,1,1,1])
	diagonal_stack = np.column_stack((diagonal_initial[:,0],diagonal_initial[:,1],diagonal_initial[:,2],diagonal_initial[:,3],diagonal_initial[:,4],diagonal_initial[:,5]))
	diagonal_stack = diagonal_stack.reshape(-1,1,order='F')
	initial_state[6:] = diagonal_stack
	initial_state= initial_state.flatten()
	dt = 10
	t0 = 0
	tf = 10
	time = 0
	i = 1
	state_vector = np.zeros((100,6))
	state_vector[0] = y_0.flatten()
	std_devr = np.zeros((100,1))
	std_devv = np.zeros((100,1))
	std_devr[0] = (state_cov_matrix1[0,0] + state_cov_matrix1[1,1] + state_cov_matrix1[2,2])**0.5
	std_devv[0] = std_dev_v = (state_cov_matrix1[3,3] + state_cov_matrix1[4,4] + state_cov_matrix1[5,5])**0.5
	ep_array = np.zeros((100,1))
	ep_array[0] = ep
	while time < 990:
		t_span = (0, 10)
		t_eval = np.linspace(0,10,2)
		solution = solve_ivp(dydt,t_span,initial_state,t_eval=t_eval)
		t = solution.t
		y = solution.y 
		r_t = y[:3, :]  # Position vector over time
		v_t = y[3:6, :]  # Velocity vector over time
		initial_state = y[:,1:]
		Phi_t = initial_state[6:, :].reshape((6, 6)) 
		y_0 = initial_state[:6,:]
		Phi_trans = np.transpose(Phi_t)
		#defining standard deviations
		
		
		#updating state covariance
		state_cov_matrix1 = np.dot(Phi_t,np.dot(state_cov_matrix1,Phi_trans))
		
		state_cov_matrix1 += Q_k
		
		
		#update
		state_cov_matrix1,y_0,ep = updatey(obs_cov_matrix, state_cov_matrix1, y_0,i)
		ep_array[i] = ep
		std_dev_r = (state_cov_matrix1[0,0] + state_cov_matrix1[1,1] + state_cov_matrix1[2,2])**0.5
		
		std_dev_v = (state_cov_matrix1[3,3] + state_cov_matrix1[4,4] + state_cov_matrix1[5,5])**0.5
		std_devr[i] = std_dev_r
		std_devv[i] = std_dev_v
		initial_state[:6] = y_0
		state_vector[i] = y_0.flatten()
		initial_state = initial_state.flatten()
		initial_state[6:] = np.eye(6).flatten()
		
		time+=10
		i+=1
	return y, state_vector, std_devr, std_devv, ep_array


def plotting(pos, vel , file_name):
	time = tim[0:99]
	fig, axs = plt.subplots(1, 2, figsize=(10, 6)) 
	
	axs[0].plot(time, pos)
	axs[0].set_xlabel('Time(s)')
	axs[0].set_ylabel('position residuals(km)')
	

	axs[1].plot(time, vel)
	axs[1].set_xlabel('Time(s)')
	axs[1].set_ylabel('velocity residuals(km/s)')
	
	
	plt.tight_layout()
	plt.savefig(file_name)
	plt.show(block=False)
	plt.close()
	return(file_name)

#plotting standard deviations
def plotting_std(std_dev_pos, std_dev_vel, file_name):
	time = tim[0:100]
	
	fig, axs = plt.subplots(1, 2, figsize=(10, 6)) 
	
	axs[0].plot(time, std_dev_pos)
	axs[0].set_xlabel('time(s)')
	axs[0].set_ylabel('standard deviation(km)')
	

	axs[1].plot(time, std_dev_vel)
	axs[1].set_xlabel('time(s)')
	axs[1].set_ylabel('standard deviation(km/s)')
	
	
	plt.tight_layout()
	plt.savefig(file_name)
	plt.show(block=False)
	plt.close()
	return(file_name)



obs_cov = obs_cov_matrix(len(CA_range[0])-1)
state_cov_matrix1 = state_cov_matrix(0.002,0.0001,.7)
guess = np.array([[rx[0]], [ry[0]], [rz[0]], [vx[0]], [vy[0]], [vz[0]]])
print(guess)
H, delta_z = least_squares(guess, rx_gps[0], ry_gps[0], rz_gps[0], CA_range[0])
no_process_noise = np.zeros((6,6))

y, state_vector, std_dev_r, std_dev_v,ep = kalman_filter(guess, obs_cov_matrix, state_cov_matrix, no_process_noise)
process_noise = np.diag([1e-11,1e-11,1e-11,1e-9,1e-9,1e-9])
print("here")
print(state_vector)
print("hERE")
np.set_printoptions(precision=9, linewidth=200, suppress=True)
print(state_vector[9],state_vector[19],state_vector[29])
guess1 = np.array([[rx[0]], [ry[0]], [rz[0]], [vx[0]], [vy[0]], [vz[0]]])
y_p, state_process, std_dev_rp, std_dev_vp, ep_p = kalman_filter(guess1, obs_cov_matrix,state_cov_matrix, process_noise)


rx_est = state_vector[0:len(state_vector),0]
ry_est = state_vector[0:len(state_vector),1]
rz_est = state_vector[0:len(state_vector),2]
vx_est = state_vector[0:len(state_vector),3]
vy_est = state_vector[0:len(state_vector),4]
vz_est = state_vector[0:len(state_vector),5]
rx_res = abs(rx_est[0:99] - rx[0:99])
ry_res = abs(ry_est[0:99] - ry[0:99])
rz_res = abs(rz_est[0:99] - rz[0:99])
r_eucl = ((rx_res)**2 + (ry_res)**2 + (rz_res)**2)**0.5
vx_res = abs(vx_est[0:99] - vx[0:99])
vy_res = abs(vy_est[0:99] - vy[0:99])
vz_res = abs(vz_est[0:99] - vz[0:99]) 
v_eucl = ((vx_res)**2 + (vy_res)**2 + (vz_res)**2)**0.5 
rx_est_p = state_process[0:len(state_vector),0]
ry_est_p = state_process[0:len(state_vector),1]
rz_est_p = state_process[0:len(state_vector),2]
vx_est_p = state_process[0:len(state_vector),3]
vy_est_p = state_process[0:len(state_vector),4]
vz_est_p = state_process[0:len(state_vector),5]
rx_res_p = abs(rx_est_p[0:99] - rx[0:99])
ry_res_p = abs(ry_est_p[0:99] - ry[0:99])
rz_res_p = abs(rz_est_p[0:99] - rz[0:99])
r_eucl_p = ((rx_res_p)**2 + (ry_res_p)**2 + (rz_res_p)**2)**0.5
vx_res_p = abs(vx_est_p[0:99] - vx[0:99])
vy_res_p = abs(vy_est_p[0:99] - vy[0:99])
vz_res_p = abs(vz_est_p[0:99] - vz[0:99])
v_eucl_p = ((vx_res_p)**2 + (vy_res_p)**2 + (vz_res_p)**2)**0.5 
plotting(r_eucl, v_eucl  ,'posvel_res.png')
plotting(r_eucl_p, v_eucl_p ,'posvel_resp.png')
plotting_std(std_dev_r, std_dev_v,'stddev.png')
plotting_std(std_dev_rp,std_dev_vp,'stddev_p.png')
print("here")
print(r_eucl_p[50:80].reshape(-1,1))
print(ep_p[50:80])


r0 = np.array([rx[0],ry[0],rz[0]])
v0 = np.array([vx[0],vy[0],vz[0]])
Phi0 = np.eye(6).flatten()
y0 = np.hstack([r0,v0,Phi0])
y0 = y0

t_span = (0, 10)
t_eval = np.linspace(0,10,2)

solution = solve_ivp(dydt,t_span,y0,t_eval=t_eval)
t = solution.t
y = solution.y 
r_t = y[:3, :]  # Position vector over time
v_t = y[3:6, :]  # Velocity vector over time
Phi_t = y[6:, :].reshape((6, 6, -1)) 
new_y = y[:,1:]
y_e = new_y[0:6]
np.set_printoptions(precision=6, linewidth=200, suppress=True)

phit_t = new_y[6:,:].reshape((6,6))
np.set_printoptions(precision=10, linewidth=200, suppress=True)

"""
K, x = updatey(obs_cov_matrix,state_cov_matrix1, guess,0)
initial_state = np.zeros((42,1))
initial_state[:6] = x
diagonal_initial = np.diag([1,1,1,1,1,1])
diagonal_stack = np.column_stack((diagonal_initial[:,0],diagonal_initial[:,1],diagonal_initial[:,2],diagonal_initial[:,3],diagonal_initial[:,4],diagonal_initial[:,5]))
diagonal_stack = diagonal_stack.reshape(-1,1,order='F')
initial_state[6:] = diagonal_stack



t0 = 0          # Initial time
t_final = 10    # Final time
dt = 10        # Time step

# Create the ODE solver
solver = ode(integrator).set_integrator('dopri5')
solver.set_initial_value(initial_state, t0)

# Propagate the system
times = []
combined_states = []


while solver.successful() and solver.t < t_final:
    solver.integrate(solver.t + dt)
    times.append(solver.t)
    combined_states.append(solver.y)

# Convert results to arrays for easier handling
times = np.array(times)
combined_states = np.array(combined_states)

# Extract final results
final_phi = combined_states[-1, 6:].reshape(6, 6)  # Final state transition matrix
final_state_vector = combined_states[-1, :6]   # First 6 elements
"""








