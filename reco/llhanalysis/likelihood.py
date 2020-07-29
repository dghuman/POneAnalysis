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
from iminuit import Minuit
import argparse
import math

# Some quantities that are environment dependent
c = 2.99792458e8 * I3Units.m / I3Units.second   # speed of light 
n = 1.34                                        # 1.33 is the refractive index of water at 20 degrees C
c_m = c/n                                       # light in water
theta_c = np.arccos(1./n)                       # Cherenkov angle in water
lambda_s = 120                                  # scattering length of light for violet light
lambda_a = 15.                                   # absorption length of light for violet light
tau = 557E-9                                    # time parameter that has to be fit using simulations or data      

# DISCLAIMER: lambda_s, lambda_a and tau all need to be fit using data or simulation for water. The values here are from ICECUBE and aren't the same for water.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Creates a reconstruction of the muon track using a likelihood pdf.")
    parser.add_argument( '-H', '--hitThresh', default = 1, help = "threshold of hits for the DOM to be considered")
    parser.add_argument( '-D', '--domThresh', default = 7, help = "threshold of hit DOMs for the frame to be considered")
    parser.add_argument( '-R', '--maxResidual', default = 100 , help = "maximum time residual allowed for the hit to be considered")
    parser.add_argument( '-g', '--GCDType', help = "type of geometry used for the simulation set")
    parser.add_argument( '-d', '--DOMType', help = "the type of DOM used in the simulation")
    args = parser.parse_args()
                                                                                                                                                       
# The likelihood is only as well defined as it's PDF. We use the Pandel Function here for that. Takes t and d as numpy arrays 
# Note that since one of the terms in the sum is just a constant (const) we add it to the initial term for now
# We make this an object so it can be passed as an argument. This also means we can introduce other pdf's if we want to
def pdfPandel():    
    def Pandel(t, d, a_lambda = lambda_a, s_lambda = lambda_s, Tau = tau):
        const = np.ones(d.size)*(1. + Tau*c_m/a_lambda)
        first = (d/s_lambda)*np.log(t/Tau)
        second = np.log(t*sp.gamma(d/s_lambda))
        third = t*((1./Tau) + c_m/a_lambda)
        result = -1.*(first - second - third) + const
        return result
    return Pandel

# Pulled from dvir_Analysis/improvedTrackReco.py. Used to find the first hit time
def get_t_0(frame):
    mcpeMap = frame["MCPESeriesMap_significant_hits"]
    mcpeList = mcpeMap[mcpeMap.keys()[0]]
    return mcpeList[0].time

# Pulled directly from dvir_Analysis/imporvedTrackReco.py. Just a first guess using linefit. t_0 is the first hit time for each DOM
def InitialGuess(frame, domsUsed, t_0, geometry):
    data = SimAnalysis.getLinefitDataPoints(frame, geometry)
    u, speed, vertex = SimAnalysis.linefitParticleParams(data)

    phi = np.arctan2(u.y, u.x)/I3Units.deg 
    if phi < 0:
        phi += 360

    theta = np.arcsin(u.z)/I3Units.deg
    
    linefit = dataclasses.I3Particle()
    linefit.shape = dataclasses.I3Particle.InfiniteTrack
    linefit.fit_status = dataclasses.I3Particle.OK
    linefit.dir = u
    linefit.speed = speed
    linefit.pos = vertex
    linefit.time = 0

    delta_q = dataclasses.I3Position(u.x*speed*t_0, u.y*speed*t_0, u.z*speed*t_0)

    q = vertex + delta_q

    return q.x, q.y, q.z, theta, phi, linefit, data

# Functional that is fed data from InitialGuess for PMT locations and the PDF we wish to use. Uses those locations to build a Pandel Function for a given track
def LikelihoodFunctor(data, pdf):
    # turn PMT locations and time hits into numpy arrays for easier numpy algebra
    data = np.array(data)
    pmt = data[:,0:3]
    time = data[:,3]
    
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
        t_off = time[0] - d_ic/(np.sin(theta_c)*c) - (t_ic + np.sign(t_ic)*d_ic/(np.tan(theta_c)*c))
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
        t_geo = d/c
        # The total geometric time
        t_geo = t_geo + tc
        # Residual time is now the difference between the geometric time and the observed time
        t = np.absolute(t_geo - time)    # We have taken the abs() since it is easier for now than convolving with a gaussian
        return d, t

    # uses the prior defined functions to build a likelihood function that when given a track (linefit) will produce a negative loglikelihood value
    def likelihoodFunction(vx, vy, vz, theta, phi):
        dc, tc = closestApproach(vx, vy, vz, theta, phi)
        t_off = computeOffset(tc[0], dc[0])
        d, t = computeResiduals(dc, tc, t_off)
        out = pdf(t,d)
        return np.sum(out)

    return likelihoodFunction

# A comparison functor for building a function that only returns the time residual. Minimizing just the time residual can act like a comparison for the likelihood function. 
def TestFunctor(data):
    # turn PMT locations and time hits into numpy arrays for easier numpy algebra
    data = np.array(data)
    pmt = data[:,0:3]
    time = data[:,3]
    
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
        t_off = time[0] - d_ic/(np.sin(theta_c)*c) - (t_ic + np.sign(t_ic)*d_ic/(np.tan(theta_c)*c))
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
        t_geo = d/c
        # The total geometric time
        t_geo = t_geo + tc
        # Residual time is now the difference between the geometric time and the observed time
        t = np.absolute(t_geo - time)    # We have taken the abs() since it is easier for now than convolving with a gaussian
        return d, t

    # uses the prior defined functions to build a likelihood function that when given a track (linefit) will produce a negative loglikelihood value
    def likelihoodFunction(vx, vy, vz, theta, phi):
        dc, tc = closestApproach(vx, vy, vz, theta, phi)
        t_off = computeOffset(tc[0], dc[0])
        d, t = computeResiduals(dc, tc, t_off)
        return np.sum(t)

    return likelihoodFunction

# Main function of this file. Structured this way so that it can be easily imported aswell in any other implementation.                                   
def main(): 
    infile = dataio.I3File('/home/users/ghuman/ICECUBE/step3/NuMu/NuMu_F_' + str(args.GCDType) + '_141_step3.zst')
    gcdfile = dataio.I3File('/home/users/ghuman/ICECUBE/gcdfiles/' + str(args.GCDType) + '.i3')
    hitThresh = int(args.hitThresh)
    maxResidual = float(args.maxResidual)
    domThresh = int(args.domThresh)
    geometry = gcdfile.pop_frame()["I3Geometry"]
    domsUsed = geometry.omgeo.keys()
    outfile = dataio.I3File('/home/users/ghuman/simAnalysis/output/I3Files/llhfitReco/'+ str(args.GCDType) + '/NuGen_llhfitReco_' + str(args.GCDType) + '_d' + str(args.domThresh) + '_141.i3.gz', 'w')
    printOutFile = open('MinimizerOutput.txt','w')

    for frame in infile:
        if SimAnalysis.passFrame(frame, domsUsed, hitThresh, domThresh, maxResidual, geometry.omgeo):
            frame = SimAnalysis.writeSigHitsMapToFrame(frame, domsUsed, hitThresh, domThresh, maxResidual, geometry.omgeo)
            t_0 = get_t_0(frame)
            initialGuess = InitialGuess(frame, domsUsed, t_0, geometry)
            # Now we call our likelihoodfunctor using our pdf object. initialGuess[6] == data == pmt info
            qFunctor = LikelihoodFunctor(initialGuess[6], pdfPandel())  
            # functor for time
            qTFunctor = TestFunctor(initialGuess[6])                    
            minimizer = Minuit(qFunctor, vx = initialGuess[0], vy = initialGuess[1], vz = initialGuess[2], 
                               theta = initialGuess[3], phi = initialGuess[4], error_vx = 1, error_vy = 1, 
                               error_vz = 1, error_theta = np.arcsin(0.05)/I3Units.deg, error_phi = 1, errordef = 1,
                               limit_theta = (-90.0, 90.0), limit_phi = (0, 360) )
            minimizer.migrad()

            # Minimize time
            Tminimizer = Minuit(qTFunctor, vx = initialGuess[0], vy = initialGuess[1], vz = initialGuess[2], 
                               theta = initialGuess[3], phi = initialGuess[4], error_vx = 1, error_vy = 1, 
                               error_vz = 1, error_theta = np.arcsin(0.05)/I3Units.deg, error_phi = 1, errordef = 1,
                               limit_theta = (-90.0, 90.0), limit_phi = (0, 360) )
            Tminimizer.migrad()

            # record minimizer results in output file
            # for likelihood
            printOutFile.write("Likelihood results:\n")
            printOutFile.write(str(minimizer.get_fmin()) + '\n')
            printOutFile.write( str(minimizer.values) + '\n')
            printOutFile.write('\n\n')
            # for time
            printOutFile.write("Time results:\n")
            printOutFile.write(str(Tminimizer.get_fmin()) + '\n')
            printOutFile.write( str(Tminimizer.values) + '\n')
            printOutFile.write('\n\n')

            Tsolution = Tminimizer.values
            solution = minimizer.values
            
            # For likelihood
            q = dataclasses.I3Position(solution['vx'], solution['vy'], solution['vz'])
            phi = solution['phi'] * I3Units.deg
            theta = solution['theta'] * I3Units.deg
            u = dataclasses.I3Direction(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.sin(theta))

            # For time
            Tq = dataclasses.I3Position(Tsolution['vx'], Tsolution['vy'], Tsolution['vz'])
            Tphi = Tsolution['phi'] * I3Units.deg
            Ttheta = Tsolution['theta'] * I3Units.deg
            Tu = dataclasses.I3Direction(np.sin(Ttheta)*np.cos(Tphi), np.sin(Ttheta)*np.sin(Tphi), np.sin(Ttheta))

            # Record the final result
            recoParticle = dataclasses.I3Particle()
            recoParticle.shape = dataclasses.I3Particle.InfiniteTrack

            TrecoParticle = dataclasses.I3Particle()
            TrecoParticle.shape = dataclasses.I3Particle.InfiniteTrack

            # record on particle whether reconstruction was successful
            if minimizer.get_fmin()["is_valid"]:
                recoParticle.fit_status = dataclasses.I3Particle.OK
            else:
                recoParticle.fit_status = dataclasses.I3Particle.InsufficientQuality
                    
            if Tminimizer.get_fmin()["is_valid"]:
                TrecoParticle.fit_status = dataclasses.I3Particle.OK
            else:
                TrecoParticle.fit_status = dataclasses.I3Particle.InsufficientQuality

            recoParticle.dir = u
            recoParticle.speed = c
            recoParticle.pos = q
            recoParticle.time = 0
        
            TrecoParticle.dir = Tu
            TrecoParticle.speed = c
            TrecoParticle.pos = Tq
            TrecoParticle.time = 0

            # include both linefit and improved recos for comparison
            frame.Put('LlhFitRecoParticle', recoParticle)
            frame.Put('TimeFitRecoParticle', TrecoParticle)
            frame.Put('LineFitRecoParticle', initialGuess[5])
            outfile.push(frame)
                    
    outfile.close()
    printOutFile.close()
    return 0
            
if __name__ == '__main__':                                                                                          
    main()                                                                                                                                               

        


