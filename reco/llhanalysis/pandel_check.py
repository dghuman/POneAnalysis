#!/usr/bin/env python
# A script meant to help understand what the pandel function does. Moreover, it can be used to help us know how to modify the function to best fit our uses. 

from likelihood import pdfPandel
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from scipy import special as sp

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

# comparison pandel function
def test_pandel(t,d,lambda_a = 15., lambda_s = 120., tau = 557E-9):
    N = np.exp(-d/lambda_a)*np.power((1. + (tau*c_m)/lambda_a),-d/lambda_s)
    exp = np.exp(-t*((1./tau)+(c_m/lambda_a))-d/lambda_a)
    frac = (np.power(tau,-d/lambda_s)*np.power(t,(d/lambda_s) - 1))/(sp.gamma(d/lambda_s))
    return -np.log((1./N)*frac*exp)


t = np.linspace(1E-9, 500E-9, n)
d = np.linspace(1, 100, n)

time_dist_1 = np.zeros((n,n))
time_dist_2 = np.zeros((n,n))


if __name__ == '__main__':
    
    for i in range(n):
        test_t = np.ones(n)*t[i]
        time_dist_1[:,i] = pandel(test_t,d)
        time_dist_2[:,i] = test_pandel(test_t,d)
        
    fig = plt.figure(1)
    final = np.abs(time_dist_1 - time_dist_2)
    im = plt.imshow(final, cmap=plt.get_cmap('hot'), vmin=0, vmax = 10)
    plt.title('Difference between methods')
    plt.xticks(range(0,100,20), [i*(4.99) + 1 for i in range(0,100,20)])
    plt.yticks(range(0,100,20), [i*(0.99) + 1 for i in range(0,100,20)])
    plt.xlabel(r'Time (ns)')
    plt.ylabel(r'Distance (m)')
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel(r'$\ell$', labelpad=5)
    fig.colorbar(im,cbar_ax)
    plt.savefig('/home/users/ghuman/simAnalysis/output/plots/llhPDF/2D_pandel_comp.png', dpi=300)
