#!/usr/bin/env python                                                                                                                                    
# This is meant to be a slightly more robust approach to reconstruction of a muon event.                                                                 
# The physics and likelihood model is heavily based off of the ICECUBE model and can be found at                                                         
# "https://publications.ub.uni-mainz.de/theses/volltexte/2014/3869/pdf/3869.pdf"                                                                          

# Import some useful ICECUBE modules                                                                                                                      

from icecube import dataclasses, dataio, simclasses                                                                                                      
from icecube.icetray import I3Units, I3Frame                                                                                                             
from icecube.dataclasses import I3Particle                                                                                                               
import numpy as np                                                                                                                                     
from dvir_Analysis import SimAnalysis                  # This modules contains everything we need to do linefit for our prior                            
from scipy import special as sp                        # For the Gamma function                                                                        

# Some quantities that are environment dependent
c = 2.99792458e8 * I3Units.m / I3Units.second   # speed of light 
n = 1.34                                        # 1.33 is the refractive index of water at 20 degrees C
c_medium = c/n                                  # light in water
lambda_s = 33.3 * I3Units.m                     # scattering length of light
lambda_a = 98 * I3Units.m                       # absorption length of light
tau = 557E-9 * I3Units.second                   # not sure what this is...      CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                                                                                                                       
# The likelihood is only as well defined as it's PDF. We use the Pandel Function here for that. Takes t_res and d as numpy arrays 
def Pandel(t_res, d):
    d_l = np.divide(d,lambda_s)
    N = np.exp(-np.divide(d,lambda_a))*np.power(1. + (tau * c_medium)/lambda_a, -d_l)
    gamma = sp.gamma(d_l)
    num = np.power(tau, -d_l) * np.power(t_res, d_l - 1.)
    exp = np.exp(-t_res * (1./tau + c_medium/lambda_a) - np.divide(d, lambda_a))
    return (1/N)*(num/gamma)*exp

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

# Functional that is fed data from InitialGuess for PMT locations. Uses those locations to build a Pandel Function for a given track
class LikelihoodFunctor():
    
    def __init__(self, frame, t, d):
        self.frame = frame 
        self.t = t
        self.d = d
        
    def likelihoodFunction(self, linefit):
        


# Main function of this file. Structured this way so that it can be easily imported aswell in any other implementation.                                   

def main():                                                                                                                                             
 
if __name__ = '__main__':                                                                                          
    main()                                                                                                                                               

        
