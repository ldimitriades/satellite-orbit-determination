import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt('t.txt') # epochs - s

t_f = t - t[0]

CA_range = np.loadtxt('CA_range.txt') # pseudorange observations from CA code - km
PRN_ID = np.loadtxt('PRN_ID.txt') # PRN ID of tracked GPS satellites

print(PRN_ID[49])
clk_gps = np.loadtxt('clk_gps.txt') # clock correction for GPS sats (transmitters) - s

rx_gps = np.loadtxt('rx_gps.txt') # GPS satellite positions (transmitters) - km
ry_gps = np.loadtxt('ry_gps.txt')
rz_gps = np.loadtxt('rz_gps.txt')


vx_gps = np.loadtxt('vx_gps.txt') # GPS satellite velocities (transmitters) - km/s
vy_gps = np.loadtxt('vy_gps.txt')
vz_gps = np.loadtxt('vz_gps.txt')
#print(vx_gps,vy_gps,vz_gps)
rx = np.loadtxt('rx.txt') # precise positions (receivers) - km
ry = np.loadtxt('ry.txt')
rz = np.loadtxt('rz.txt')
vx = np.loadtxt('vx.txt') # precise velocities (receivers) - km/s
vy = np.loadtxt('vy.txt')
vz = np.loadtxt('vz.txt')
#print(vx,vy,vz)


print(rx_gps[0])
mu = 398600
c = (299792458)/1000 #km/s
E = np.arange(0,6, 0.1, dtype = float)
a = 26560
e = 0.01
r_rel = -(2/c)*(a*e*np.sin(E)*(mu/a)**.5)
plt.plot(E, r_rel)
plt.xlabel('E(rad)')
plt.ylabel('position(km)')
plt.tight_layout()
plt.savefig('rel.png')
plt.show()
#initial guess 

def least_squares(x_r,y_r,z_r,t_r, rx_gps, ry_gps, rz_gps, clk_gps,CA_range, size):
	i=0
	obs = np.trim_zeros(CA_range)
	f_x_matrix = np.zeros((1, size))
	par_x_matrix = np.zeros((1, size))
	par_y_matrix = np.zeros((1, size))
	par_z_matrix = np.zeros((1, size))
	par_clock_matrix = np.zeros((1, size))
	while(i<size):
			f_x = ((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5 + c*(t_r - clk_gps[i])
			part_rx_receiver = (x_r - rx_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			part_ry_receiver = (y_r - ry_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			part_rz_receiver = (z_r - rz_gps[i]) / (((x_r - rx_gps[i])**2 + (y_r - ry_gps[i])**2 + (z_r - rz_gps[i])**2)**.5)
			part_time_receiver = c
			f_x_matrix[0,i] = f_x.item()
			par_x_matrix[0,i] = part_rx_receiver.item()
			par_y_matrix[0,i] = part_ry_receiver.item()
			par_z_matrix[0,i] = part_rz_receiver.item()
			par_clock_matrix[0,i] = part_time_receiver
			i+=1

	par_x_matrix = par_x_matrix.reshape(size,1)
	par_y_matrix = par_y_matrix.reshape(size,1)
	par_z_matrix = par_z_matrix.reshape(size,1)
	par_clock_matrix = par_clock_matrix.reshape(size,1)
	comb_matrix = np.hstack([par_x_matrix,par_y_matrix,par_z_matrix,par_clock_matrix])
	delta_y = obs - f_x_matrix
	delta_y = delta_y.reshape(size,1)
	return comb_matrix, delta_y


b = np.trim_zeros(rx_gps[0])
#all this stays constant
def weight_matrix(size):
    sigma = 0.003
    correlation = 0.2
    cov_matrix = np.zeros((size, size))
    for l in range(size):
        for m in range(size):
            if l == m:
                cov_matrix[l, m] = sigma ** 2
            else:
                cov_matrix[l, m] = correlation * (sigma ** 2)

    W_matrix = (np.linalg.inv(cov_matrix)) / sigma**2
    return cov_matrix, W_matrix

def iter_f(rx_gps, ry_gps, rz_gps, clk_gps, CA_range):
	#input should be just for the one epoch
	CA_range = np.trim_zeros(CA_range)
	size = len(CA_range)
	rx_gps = np.trim_zeros(rx_gps)
	ry_gps = np.trim_zeros(ry_gps)
	rz_gps = np.trim_zeros(rz_gps)
	clk_gps = np.trim_zeros(clk_gps)
	#define initial guess
	x_r = 3000
	y_r = 3000
	z_r  = 3000
	t_r_off = 0.08
	k=0
	while(k<6):
		H, delta_y = least_squares(x_r, y_r, z_r, t_r_off, rx_gps, ry_gps, rz_gps, clk_gps, CA_range, size) 
		covariance, W = weight_matrix(size)
		H_t = np.transpose(H)
		N = np.dot(np.dot(H_t,W),H)
		b = np.dot(np.dot(H_t,W),delta_y)
		dx = np.linalg.solve(N, b)
		x_r += dx[0]
		y_r += dx[1]
		z_r += dx[2]
		t_r_off += dx[3]
		k += 1
	return x_r, y_r, z_r, t_r_off, delta_y

def all_epoch(rx_gps, ry_gps, rz_gps, clk_gps, CA_range):
	num = len(CA_range)
	x_r_all = np.zeros((1, num))
	y_r_all = np.zeros((1, num))
	z_r_all = np.zeros((1, num))
	t_r_all = np.zeros((1, num))
	n=0
	
	while(n<num):
		x_r, y_r, z_r, t_r_off, delta_y = iter_f(rx_gps[n], ry_gps[n], rz_gps[n], clk_gps[n], CA_range[n])
		x_r_all[0,n] = x_r.item()
		y_r_all[0,n] = y_r.item()
		z_r_all[0,n] = z_r.item()
		t_r_all[0,n] = t_r_off.item()
		n+=1
	return x_r_all, y_r_all, z_r_all, t_r_all, delta_y

def receiver_clk_corr(t_r):
	n = len(rx)
	rx_corr = np.zeros((n))
	ry_corr = np.zeros((n))
	rz_corr = np.zeros((n))
	for i in range(n):
		rx_corr[i] = rx[i] - t_r[0,i]*vx[i]
		ry_corr[i] = ry[i] - t_r[0,i]*vy[i]
		rz_corr[i] = rz[i] - t_r[0,i]*vz[i]

	return rx_corr, ry_corr, rz_corr

def plottting(epoch,x,y,z, file_name):
	time = epoch
	fig, axs = plt.subplots(1, 3, figsize=(10, 6)) 
	
	axs[0].plot(time, x)
	axs[0].set_xlabel('Time(hours)')
	axs[0].set_ylabel('x-residuals(km)')
	

	axs[1].plot(time, y)
	axs[1].set_xlabel('Time(hours)')
	axs[1].set_ylabel('y-residuals(km)')
	

	axs[2].plot(time, z)
	axs[2].set_xlabel('Time(hours)')
	axs[2].set_ylabel('z-residuals(km)')
	
	plt.tight_layout()
	plt.savefig(file_name)
	plt.show(block=False)
	plt.close()
	return(file_name)



x_r_, y_r_,z_r_,t_r_, delta_y1 = all_epoch(rx_gps, ry_gps, rz_gps, clk_gps, CA_range)
print(x_r_)
print(y_r_)
print("z")
print(z_r_)
rx_corr_1, ry_corr_1, rz_corr_1 = receiver_clk_corr(t_r_)
rx_corr_1 = rx_corr_1.reshape(-1,1)
ry_corr_1 = ry_corr_1.reshape(-1,1)
rz_corr_1 = rz_corr_1.reshape(-1,1)
x_r_ = x_r_.reshape(-1,1)
res_x = x_r_ - rx_corr_1
y_r_ = y_r_.reshape(-1,1)
res_y = y_r_ - ry_corr_1
z_r_ = z_r_.reshape(-1,1)
res_z = z_r_ - rz_corr_1

epoch  = np.arange(1,201, 1, dtype = int)

plottting(epoch, res_x, res_y, res_z, 'uncorrected.png')



def light_time_correction(CA_range, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps):
     # Speed of light in m/s
    epochs = len(CA_range)
    
    # Determine the maximum number of satellites across epochs
    max_satellites = max(len(np.trim_zeros(CA_range[n])) for n in range(epochs))
    
    # Preallocate arrays with NaN values to handle varying lengths
    r_x_lt = np.zeros((epochs, max_satellites))
    r_y_lt = np.zeros((epochs, max_satellites))
    r_z_lt = np.zeros((epochs, max_satellites))

    for n in range(epochs):
        p = len(np.trim_zeros(CA_range[n]))  # Number of satellites at epoch `n`
        
        for i in range(p):
            rx = np.trim_zeros(rx_gps[n])
            ry = np.trim_zeros(ry_gps[n])
            rz = np.trim_zeros(rz_gps[n])
            tl = CA_range[n, i] / c
            lt_x = tl * vx_gps[n, i]
            lt_y = tl * vy_gps[n, i]
            lt_z = tl * vz_gps[n, i]
            cor_x = rx[i] - lt_x
            cor_y = ry[i] - lt_y
            cor_z = rz[i] - lt_z
            cor_r = np.array([cor_x, cor_y, cor_z]).reshape(3, 1)

            ang_vel = 7.292115e-5  # Earth's angular velocity in rad/s
            phi = ang_vel * tl

            R_z = np.array([
                [np.cos(phi), np.sin(phi), 0],
                [-np.sin(phi), np.cos(phi), 0],
                [0, 0, 1]
            ])
            cor_ang = np.dot(R_z, cor_r).flatten()  # Flatten to a 1D array

            
            r_x_lt[n, i] = cor_ang[0]
            r_y_lt[n, i] = cor_ang[1]
            r_z_lt[n, i] = cor_ang[2]

    return r_x_lt, r_y_lt, r_z_lt

rx_gps_cor, ry_gps_cor, rz_gps_cor = light_time_correction(CA_range, rx_gps,ry_gps,rz_gps,vx_gps,vy_gps,vz_gps)
print("here")
print(rx_gps[0])
print(rx_gps_cor[0])
def rel_correction():
	epochs = len(CA_range)
	max_satellites = max(len(np.trim_zeros(CA_range[n])) for n in range(epochs))
	CA_range_cor = np.zeros((epochs, max_satellites))
	for n in range(epochs):
		p = len(np.trim_zeros(CA_range[n]))

		for i in range(p):
			rx = np.trim_zeros(rx_gps[n])
			ry = np.trim_zeros(ry_gps[n])
			rz = np.trim_zeros(rz_gps[n])
			vx = np.trim_zeros(vx_gps[n])
			vy = np.trim_zeros(vy_gps[n])
			vz = np.trim_zeros(vz_gps[n])

			r = np.array([rx[i],ry[i],rz[i]])
			v = np.array([vx[i],vy[i],vz[i]])
			x = (2 / c)*np.dot(r, v)
			
			CA_range_cor[n,i] = CA_range[n,i] - (2 / c)*np.dot(r, v)

	return CA_range_cor

ca_range_cor = rel_correction()

x_r_cor, y_r_cor, z_r_cor, t_r_cor, delta_y2 = all_epoch(rx_gps_cor, ry_gps_cor, rz_gps_cor, clk_gps, ca_range_cor)
print(x_r_cor)
print(y_r_cor)
print(z_r_cor)
rx_corr_2, ry_corr_2, rz_corr_2 = receiver_clk_corr(t_r_cor)
eps_cor = - delta_y2
rx_corr_2 = rx_corr_2.reshape(-1,1)
ry_corr_2 = ry_corr_2.reshape(-1,1)
rz_corr_2 = rz_corr_2.reshape(-1,1)
x_r_cor = x_r_cor.reshape(-1,1)
res_x_cor = x_r_cor - rx_corr_2  
y_r_cor = y_r_cor.reshape(-1,1)
res_y_cor = y_r_cor - ry_corr_2
z_r_cor = z_r_cor.reshape(-1,1)
res_z_cor = z_r_cor - rz_corr_2 

plottting(epoch, res_x_cor, res_y_cor, res_z_cor, "corrected.png")
r_corr = (res_x_cor**2 + res_y_cor**2 + res_z_cor**2)**.5 

pdop = (0.003**2 + 0.003**2 + 0.003**2)**.5
r_res = (res_x_cor**2 + res_y_cor**2 + res_z_cor**2)**.5
pdop_array = np.full(200, pdop)
plt.plot(epoch, pdop_array, label='PDOP(km)')
plt.plot(epoch, r_res, label='Residuals(km)')
plt.xlabel("Epoch")
plt.legend()
plt.savefig('pdop.png')
plt.show()
t_r_cor = t_r_cor.reshape(-1,1)

plt.plot(epoch, t_r_cor)
plt.xlabel('Epoch')
plt.ylabel('receiver clock error(s)')
plt.tight_layout()
plt.savefig('timecorr.png')
plt.show(block=False)
