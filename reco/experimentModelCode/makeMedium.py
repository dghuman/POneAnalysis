#!/usr/bin/env python

'''
uses properties from "Transmission of light in deep sea water at the
site of the Antares neutrino telescope" Paper (in 
propagationMediumModels/documentation/ANTARTESWaterProperties.pdf)
to create the files clsim takes to construct the propagation medium
'''

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from simAnalysis.SimAnalysis import fitLeastSquaresLine

# change directory to the proper model directory!!
outfilepath = "/home/dvir/workFolder/P_ONE_dvirhilu/propagationMediumModels/STRAW/"

# ANTARES model data June 1999
#eda = 0.17
#effScatteringCoeff = np.array([1/265.0, 1/119.0])
#absorptionCoeff = np.array([1/68.6, 1/23.5])
#wavelengths = np.array([473, 375])

# STRAW model data April 2019
eda = 0.6
effScatteringCoeff = np.array([1/52.5, 1/81.3])
absorptionCoeff = np.array([1/28.8, 1/38.7])
wavelengths = np.array([405, 465])

# configuration  
header = '# configuration file\n'
domRadius = str(5) + '\t# over-R: DOM radius "oversize" scaling factor\n'               
efficiencyCorr = str(1.0) + '\t# overall DOM efficiency correction\n'
functionShape = str(0.0) + '\t# 0=HG; 1=SAM (scattering function shape parameter)\n'
averageAngle = str((1-eda)*0.924) + '\t# g=<cos(theta)> (formula from ANTARES paper)\n'

configArray = [header, domRadius, efficiencyCorr, functionShape, averageAngle]

outfile = open(outfilepath + "cfg.txt", 'w')

for line in configArray:
    outfile.write(line)

outfile.close()

# parameters
# least squares fit to find alpha and kappa
scatLSS = fitLeastSquaresLine(np.log(wavelengths/400.0), np.log(effScatteringCoeff.T))
absLSS = fitLeastSquaresLine(np.log(wavelengths/400.0), np.log(absorptionCoeff.T))

alpha = str(-scatLSS[0]) + '\t' + str(0) + '\t# scattering wavelength dependence power law exponent\n'
kappa = str(-absLSS[0]) + '\t' + str(0) + '\t# absorption wavelength dependence power law exponent\n'
b_e_400 = np.e**scatLSS[1]
a_e_400 = np.e**absLSS[1]


# check fit works
x = np.linspace(300,500,201)
y = [ (b_e_400)*( (i/400)**scatLSS[0])  for i in x]
plt.plot(x,y)
plt.scatter( wavelengths, effScatteringCoeff, label = r'$\alpha:\,$' + str(scatLSS[0]) + '\n' + r'$b_{eff}(400):\,$' + str(b_e_400) + r'$\,m^{-1}$' + '\n' + r'$\lambda_{sct}^{eff}(400):\,$' + str(1/b_e_400)+ r'$\,m$')
plt.title("Scattering Coefficient Wavelength Dependence")
plt.xlabel("Wavelength" r'$(nm)$')
plt.ylabel("Effective Scattering Coefficient " r'$(m^{-1})$')
plt.legend(borderpad = 1.5)

# check fit works
plt.figure()
x = np.linspace(300,500,201)
y = [ (a_e_400)*( (i/400)**absLSS[0])  for i in x]
plt.plot(x,y)
plt.scatter( wavelengths, absorptionCoeff, label = r'$\kappa:\,$' + str(absLSS[0]) + '\n' + r'$a(400):\,$' + str(a_e_400) + r'$\,m^{-1}$' + '\n' + r'$\lambda_{abs}(400):\,$' + str(1/a_e_400) + r'$\,m$')
plt.title("Absorption Coefficient Wavelength Dependence")
plt.xlabel("Wavelength " r'$nm$')
plt.ylabel("Absorption Coefficient " + r'$m^{-1}$')
plt.legend(borderpad = 1.25)

plt.show()

# keep only power law portion of absorption coefficient
A = str(0) + '\t' + str(0) + '\t# absorption exponential component multiple - ignored\n'
B = str(0) + '\t' + str(0) + '\t# absorption exponential component exponent multiple - ignored\n'

configArray = [alpha, kappa, A, B]

outfile = open(outfilepath + "icemodel.par", 'w')

for line in configArray:
    outfile.write(line)

outfile.close()

# writing data: assumes all absorption falls under power law component
min_depth = 1000
max_depth = 2800
spacing = 10
num_elements = int(((max_depth - min_depth) / spacing)) + 1
depth = np.linspace(1000,2800,num_elements)
b_e = [b_e_400 for i in xrange(num_elements)]
a_e = [a_e_400 for i in xrange(num_elements)]
# assume no temperature difference
dT = [0 for i in xrange(num_elements)]

outfile = open(outfilepath + "icemodel.dat", 'w')

for i in xrange(num_elements):
    line = str(depth[i]) + " " + str(b_e[i]) + " " + str(a_e[i]) + " " + str(dT[i]) + '\n'
    outfile.write(line)

outfile.close()
