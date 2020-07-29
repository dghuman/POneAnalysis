#!/usr/bin/env python
# A script meant to help understand what the pandel function does. Moreover, it can be used to help us know how to modify the function to best fit our uses. 

from likelihood import pdfPandel
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Some quantities that are environment dependent
c = 2.99792458e8                                # speed of light 
n = 1.34                                        # 1.33 is the refractive index of water at 20 degrees C
c_m = c/n                                       # light in water
theta_c = np.arccos(1./n)                       # Cherenkov angle in water
#lambda_s = 120 * I3Units.m                     # scattering length of light for violet light
#lambda_a = 15 * I3Units.m                      # absorption length of light for violet light
#tau = 557E-9 * I3Units.second                  # time parameter that has to be fit using simulations or data      
n = 100                                         # Number of steps

# calling pdfPandel returns a function, so we only need to call it once to make our actual PDF
pandel = pdfPandel()

# Now we make the np arrays we will be plotting over. 
lambda_a = np.linspace(5, 30, n)
lambda_s = np.linspace(100, 150, n)
tau = np.linspace(100E-9, 100E-5, n)
t = np.linspace(1E-9, 500E-9, n)
d = np.linspace(1, 100, n)

# First let's make a distance and time plot in two dimensions as a heat map. 
time_dist = np.zeros((n,n))
time_lambdas = np.zeros((n,n))
time_lambdaa = np.zeros((n,n))
time_tau = np.zeros((n,n))

for i in range(n):
    ones = np.ones(n)
    t_test = t[i]*ones
    d_test = d[i]*ones
    t_fix = (10E-9)*ones
    time_dist[:,i] = pandel(t_test, d, 10, 130, 557E-9)
    time_lambdaa[:,i] = pandel(t_fix, d_test, lambda_a, 130, 557E-9)
    time_lambdas[:,i] = pandel(t_fix, d_test, 10, lambda_s, 557E-9)
    time_tau[:,i] = pandel(t_fix, d_test, 10, 130, tau)

fig = plt.figure(1)
im = plt.imshow(time_dist, cmap=plt.get_cmap('hot'), vmin=-20, vmax=-10)
plt.title('Likelihood Distribution')
plt.xticks(range(0,100,20), [i*(4.99) + 1 for i in range(0,100,20)])
plt.yticks(range(0,100,20), [i*(0.99) + 1 for i in range(0,100,20)])
plt.xlabel(r'Time (ns)')
plt.ylabel(r'Distance (m)')
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel(r'$\ell$', labelpad=5)
fig.colorbar(im,cbar_ax)
plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/2D_time_distance_pandel.png', dpi=300)

plt.clf()
im = plt.imshow(time_lambdaa, cmap=plt.get_cmap('hot'), vmin=-20, vmax=-10)
plt.title('Likelihood Distribution')
plt.xticks(range(0,100,20), [i*(0.99) + 1 for i in range(0,100,20)])
plt.yticks(range(0,100,20), [i*(25./n) + 5 for i in range(0,100,20)])
plt.xlabel(r'Distance (m)')
plt.ylabel(r'$\lambda_a$ (m)')
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel(r'$\ell$', labelpad=5)
fig.colorbar(im,cbar_ax)
plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/2D_distance_lambda_absorp_pandel.png', dpi=300)

plt.clf()
im = plt.imshow(time_lambdas, cmap=plt.get_cmap('hot'), vmin=-20, vmax=-10)
plt.title('Likelihood Distribution')
plt.xticks(range(0,100,20), [i*(0.99) + 1 for i in range(0,100,20)])
plt.yticks(range(0,100,20), [i*(50./n) + 100 for i in range(0,100,20)])
plt.xlabel(r'Distance (m)')
plt.ylabel(r'$\lambda_s$ (m)')
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel(r'$\ell$', labelpad=5)
fig.colorbar(im,cbar_ax)
plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/2D_distance_lambda_scattering_pandel.png', dpi=300)

plt.clf()
im = plt.imshow(time_tau, cmap=plt.get_cmap('hot'), vmin=-20, vmax=-10)
plt.title('Likelihood Distribution')
plt.xticks(range(0,100,20), [i*(0.99) + 1 for i in range(0,100,20)])
plt.yticks(range(0,100,20), [i*(100E-9 - 100E-5) + 1 for i in range(0,100,20)])
plt.xlabel(r'Distance (m)')
plt.ylabel(r'$\tau$ (m)')
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel(r'$\ell$', labelpad=5)
fig.colorbar(im,cbar_ax)
plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/2D_distance_tau_pandel.png', dpi=300)

plt.clf()
plt.plot(t, time_dist[5,:], '#FF0000', label='d = ' + str(round(d[5])))
plt.plot(t, time_dist[25,:], '#00FF55', label='d = ' + str(round(d[25])))
plt.plot(t, time_dist[50,:], '#C600FF', label='d = ' + str(round(d[50])))
plt.plot(t, time_dist[75,:], '#00F0FF', label='d = ' + str(round(d[75])))
plt.legend()
plt.xlabel(r'time$(s)$')
plt.ylabel(r'-loglikelihood')
plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/distance_comp.png', dpi=300)

