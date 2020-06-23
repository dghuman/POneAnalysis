#!/usr/bin/env python                                                                                                                                    
# This is meant to be a slightly more robust approach to reconstruction of a muon event.                                                                 
# The physics and likelihood model is heavily based off of the ICECUBE model and can be found at                                                         
# "https://publications.ub.uni-mainz.de/theses/volltexte/2014/3869/pdf/3869.pdf"                                                                          
# The time residuals are computed by myself though. The techniques used are detailed in a text document I have somewhere -dg

# Import some useful ICECUBE modules                                                                                                                      

from icecube import dataclasses, dataio, simclasses                                                                                                      
from icecube.icetray import I3Units, I3Frame                                                                                                             
from icecube.dataclasses import I3Particle                                                                                                               
import numpy as np                                                                                                                                     
from dvir_Analysis import SimAnalysis                  # This modules contains everything we need to do linefit for our prior                            
from scipy import special as sp                        # For the Gamma function                                                                        
import sys

# Some quantities that are environment dependent
c = 2.99792458e8 * I3Units.m / I3Units.second   # speed of light 
n = 1.34                                        # 1.33 is the refractive index of water at 20 degrees C
c_m = c/n                                       # light in water
lambda_s = 33.3 * I3Units.m                     # scattering length of light
lambda_a = 98 * I3Units.m                       # absorption length of light
tau = 557E-9 * I3Units.second                   # time parameter that has to be fit using simulations or data      

# DISCLAIMER: lambda_s, lambda_a and tau all need to be fit using data or simulation for water. The values here are from ICECUBE and aren't the same for water.
                                                                                                                                                       
# The likelihood is only as well defined as it's PDF. We use the Pandel Function here for that. Takes t and d as numpy arrays 
# Note that since one of the terms in the sum is just a constant (const) we add it to the initial term for now
# We make this an object so it can be passed as an argument. This also means we can introduce other pdf's if we want to
def pdfPandel():    
    def Pandel(t, d):
        const = d.size*(1. + tau*c_m/lambda_a)
        first = (d/lambda_s)*np.log(t/tau)
        second = np.log(t*np.gamma(d/lambda_s))
        third = t*((1./tau) + c_m/lambda_a)
        result = -1.*(first - second - third)
        result[0] = result[0] + const
        return result
    return Pandel

# Pulled from dvir_Analysis/improvedTrackReco.py. Used to find the first hit time
def get_t_0(frame):
    mcpeMap = frame["MCPESeriesMap_significant_hits"]
    mcpeList = mcpeMap[mcpeMap.keys()[0]]
    return mcpeList[0].time

# Pulled directly from dvir_Analysis/imporvedTrackReco.py. Just a first guess using linefit. t_0 is the first hit time for each DOM
def InitialGuess(frame, domsUsed, t_0):
    data = SimAnalysis.getLinefitDataPoints(frame, geometry)
    u, speed, vertex = SimAnalysis.linefitParticleParams(data)

    phi = np.arctan2(u.y, u.x)/I3Units.deg 
    if phi < 0:
        phi += 360
    
    linefit = dataclasses.I3Particle()
    linefit.shape = dataclasses.I3Particle.InfiniteTrack
    linefit.fit_status = dataclasses.I3Particle.OK
    linefit.dir = u
    linefit.speed = speed
    linefit.pos = vertex
    linefit.time = 0

    delta_q = dataclasses.I3Position(u.x*speed*t_0, u.y*speed*t_0, u.z*speed*t_0)

    q = vertex + delta_q

    return q.x, q.y, q.z, u.z, phi, linefit, data

# Functional that is fed data from InitialGuess for PMT locations and the PDF we wish to use. Uses those locations to build a Pandel Function for a given track
def LikelihoodFunctor(data, pdf):
    # turn PMT locations and time hits into numpy arrays for easier numpy algebra
    data = np.array(data)
    pmt = data[:,0:3]
    t = data[:,3]
    
    # The computations from here on require we find the time and distance of closest approach, d_i,c and t_i,c
    def closestApproach(vx, vy, vz, theta, phi):
        # Compute vec{r} - vec{x}
        x = pmt[:,0] - vx
        y = pmt[:,1] - vy
        z = pmt[:,2] - vz
        # Compute (\vec{r} - vec{x}) dot \vec{v}
        v = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        dotprod = x*v[0] + y*v[1] + z*v[2]
        # Compute the final vector components
        x = x - dotprod*v[0]
        y = y - dotprod*v[1]
        z = z - dotprod*v[2]
        # Compute t_i,c and d_i,c
        dc = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        tc = dotprod/c
        return dc, tc

    # Time Offset used so the time of the track relates to the time observed by the first PMT 
    def computeOffset(t_ic, d_ic):
        # Pulled directly from my report
        t_off = t[0] - d_ic/(np.sin(theta_c)*c) - (t_ic + np.sign(t_ic)*d_ic/(np.tan(theta_c)*c))
        return t_off
        
    # Given the linefit and other parameters 
    def computeResiduals(dc, tc, t_off):
        # Apply our offset time to find the "true" closest approach time array
        tc = tc + t_off
        # Now we find the time of the photon emission
        tc = tc - dc/(np.tan(theta_c)*c)
        # The relative time between the first photon emission and the ith one
        tc = tc - tc[0]
        # The first component of the geometric time
        d = dc/(np.sin(theta_c)) 
        t = d/c
        # Thr total geometric time
        t = t + tc
        return d, t

    # uses the prior defined functions to build a likelihood function that when given a track (linefit) will produce a negative loglikelihood value
    def likelihoodFunction(vx, vy, vz, theta, phi):
        dc, tc = closestApproach(vx, vy, vz, theta, phi)
        t_off = computeOffset(tc[0], dc[0])
        d, t = computeResiduals(dc, tc, t_off)
        out = pdf(t,d)
        return np.sum(out)

    return likelihoodFunction

# Main function of this file. Structured this way so that it can be easily imported aswell in any other implementation.                                   
# run as shown: python ${PATH_TO_SCRIPT}/likelihood.py inputfile gcdfile hitThresh domThresh maxResidual
def main(): 
    length = len(sys.argv[1:])
    if length < 2:
        print("No Arguments given to likelihood.py")
        return 0
    infile = dataio.i3File(str(sys.argv[1]))
    gcdfile = dataio.I3File(str(sys.argv[2]))
    geometry = gcdfile.pop_frame()["I3Geometry"]
    domsUsed = geometry.omgeo.keys()
    # Some Argument parsing cause it is needed
    if length == 2:
        hitThresh = 1
        domThresh = 7
        maxResidual = 100
    elif length == 3:
        hitThresh = int(sys.argv[3])
        domThresh = 7
        maxResidual = 100
    elif length == 4:
        hitThresh = int(sys.argv[3])
        domThresh = int(sys.argv[4])
        maxResidual = 100
    else:
        hitThresh = int(sys.argv[3])
        domThresh = int(sys.argv[4])
        maxResidual = float(sys.argv[5])

    for frame in infile:
        if SimAnalysis.passFrame(frame, domsUsed, hitThresh, maxResidual, geometry.omgeo):
            frame = SimAnalysis.writeSigHitsMapToFrame(frame, domsUsed, hitThresh, domThresh, maxResidual, geometry.omgeo)
            t_0 = get_t_0(frame)
            initialGuess = calculateInitialGuess(frame, domsUsed, t_0)
            # Now we call our likelihoodfunctor using our pdf object. initialGuess[6] == data == pmt info
            qFunctor = LikelihoodFunctor(initialGuess[6], pdfPandel())
            
            
if __name__ = '__main__':                                                                                          
    main()                                                                                                                                               

        
