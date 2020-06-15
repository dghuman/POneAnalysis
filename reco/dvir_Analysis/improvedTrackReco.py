#!/usr/bin/env python

from icecube import dataclasses, dataio, icetray, simclasses
from icecube.icetray import I3Units, I3Frame
import numpy as np
import SimAnalysis
from experimentModelCode import FunctionClasses
import argparse
from iminuit import Minuit

'''
A fast track reconstruction algorithm that takes into account
the Cherenkov cone structure in its calculation (unlike linefit).
Details of the model can be found here:
https://arxiv.org/pdf/1105.4116.pdf

The reconstruction assumes the particle travels with the speed
of light in a vaccuum: p(t) = q + c*(t-t_0)*u
    q is the position vector of the particle at t=t_0
    u is the direction vector of the particle
    p(t) is the position vector of the particle at time t
    c is the speed of light in a vaccuum

Some more variables definitions for the code:
    z_c is the z component of the point of closest approach (on string)
    d_c is the horizontal distance of the point of closest approach
    t_c is the time the particle reaches its closest approach
    t_gamma is the calculated arrival time of the photon at the DOM
    d_gamma is the calculated distance of the photon to the DOM
    theta_gamma is the calculated angle of inclination of the photon w.r.t. the detector
    L_x is the x position of the DOM
    L_y is the y position of the DOM
    L_z is the z position of the DOM
    a_i is the total charge for the hit
    t_i is the hit time (minumum time in list used)
    a_0 is the chosen saturation charge (for correction function)
    sigma_i is the timing uncertainty
    d_1 is the chosen minimal distance (for correction function)

For convenience, t_0 is chosen to be 0 (linefit implementation uses t_0 = 0)
'''
# constants
c = 2.99792458e8 * I3Units.m / I3Units.second   # speed of light 
n = 1.34                                        # refractive index of water
a_0 = np.nan                                    # satruation charge (for correction function) (not used for now)
d_1 = 0.25                                      # minimal distance (for correction function) (set to just above radius of detector)
d_0 = 26.53 * I3Units.m                         # normalization distance (set to the eff. attenuation length)
sigma_i = 20 * I3Units.nanosecond               # timing uncertainty (window used is 20 ns)

parser = argparse.ArgumentParser(description = "Creates a reconstruction of the muon track using a linear least squares fit on the pulses")
#parser.add_argument( '-n', '--minFileNum', help = "smallest file number used" )
#parser.add_argument( '-N', '--maxFileNum', help = "largest file number used")
parser.add_argument( '-H', '--hitThresh', default = 1, help = "threshold of hits for the DOM to be considered")
parser.add_argument( '-D', '--domThresh', default = 7, help = "threshold of hit DOMs for the frame to be considered")
parser.add_argument( '-R', '--maxResidual', default = 100 , help = "maximum time residual allowed for the hit to be considered")
parser.add_argument( '-g', '--GCDType', help = "type of geometry used for the simulation set")
parser.add_argument( '-d', '--DOMType', help = "the type of DOM used in the simulation")
parser.add_argument( '-c', '--useMaxCharge', action = 'store_true', help = "whether to use the maximum charge saturating function")
args = parser.parse_args()
'''
if args.GCDType == 'testString':
gcdPath = '/home/dvir/workFolder/I3Files/gcd/testStrings/HorizTestString_n15_b100.0_v50.0_l1_simple_spacing.i3.gz'
elif args.GCDType == 'HorizGeo':
gcdPath = '/home/dvir/workFolder/I3Files/gcd/corHorizgeo/CorrHorizGeo_n15_b100.0_a18.0_l3_rise_fall_offset_simple_spacing.i3.gz'
elif args.GCDType == 'IceCube':
gcdPath = '/home/dvir/workFolder/I3Files/IceCube/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
elif args.GCDType == 'cube':
gcdPath = '/home/dvir/workFolder/I3Files/gcd/cube/cubeGeometry_1600_15_50.i3.gz'
else:
raise RuntimeError("Invalid GCD Type")
'''

gcdPath = '/home/users/ghuman/ICECUBE/gcdfiles/' + str(args.GCDType) + '.i3'
'''
infileList = []
for i in range(int(args.minFileNum), int(args.maxFileNum)+1):
infile = dataio.I3File('/home/dvir/workFolder/I3Files/nugen/nugenStep3/' + str(args.GCDType) + '/NuGen_step3_' + 
str(args.GCDType) + '_' + str(i) + '.i3.gz')
infileList.append(infile)
'''
infileList = [dataio.I3File('/home/users/ghuman/ICECUBE/step3/NuMu/NuMu_F_' + str(args.GCDType) + '_141_step3.zst')]
outfile = dataio.I3File('/home/users/ghuman/simAnalysis/output/I3Files/improvedfitReco/'+ str(args.GCDType) + '/NuGen_improvedfitReco_' + str(args.GCDType) + '_d' + str(args.domThresh) + '_141.i3.gz', 'w')
gcdfile = dataio.I3File(gcdPath)
geometry = gcdfile.pop_frame()["I3Geometry"]

printOutFile = open('MinimizerOutputs.txt', 'w')

# get DOM angular acceptance characteristics
inFolder = '/home/users/dhilu/P_ONE_dvirhilu/DOMCharacteristics/' + str(args.DOMType) + '/' # Will have to get my own -dg
filenameAngAcc = 'AngularAcceptance.dat'
angAcc = FunctionClasses.Polynomial(np.loadtxt(inFolder + filenameAngAcc, ndmin = 1), -1, 1)

domsUsed = geometry.omgeo.keys()
hitThresh = int(args.hitThresh)
domThresh = int(args.domThresh)
maxResidual = float(args.maxResidual)

def get_z_c(q, u, L_x, L_y):

    # multiplying an I3Position by I3Direction takes their dot product
    numerator = q.z - u.z*(q*u) + u.z*(L_x*u.x + L_y*u.y)
    denominator = 1-u.z**2

    if denominator == 0:
        return q.z

    return numerator / denominator

def get_t_c(q, u, L_x, L_y, z_c, t_0):
    return t_0 + 1/c * (L_x*u.x + L_y*u.y + z_c*u.z - q*u)

def get_d_c(q, u, t_c, L_x, L_y, t_0):
    p_x = q.x + c*(t_c-t_0)*u.x
    p_y = q.y + c*(t_c-t_0)*u.y

    return np.sqrt( (p_x - L_x)**2 + (p_y - L_y)**2 )

def get_d_gamma(u, d_c, z_c, L_z):
    ref_index_factor = n / np.sqrt(n**2 - 1)

    return ref_index_factor * np.sqrt( d_c**2 + (L_z - z_c)**2*(1 - u.z**2) )

def get_t_gamma(u, t_c, d_gamma, z_c, L_z):
    ref_index_factor = (n**2 - 1)/n

    return t_c + 1/c * (u.z*(L_z - z_c) + ref_index_factor*d_gamma)

def get_cos_theta_gamma(u, d_gamma, z_c, L_z):
    return (1 - u.z**2)*(L_z - z_c)/d_gamma + u.z/n

def getHitInformation(mcpeList):
    npeList = [mcpe.npe for mcpe in mcpeList]
    timeList = [mcpe.time for mcpe in mcpeList]

    return sum(npeList), min(timeList)

def chargeCorrectionFunction(a_i, cos_theta_gamma):
    a_iCorrected = a_i / angAcc.getValue(cos_theta_gamma) # original paper approximated a function that doesn't fit MDOM model

    if not args.useMaxCharge:
        return a_iCorrected

    numerator = a_0 * a_iCorrected
    denominator = np.sqrt(a_0**2 + a_iCorrected**2)

    return numerator/denominator

def getAverageCharge(mcpeMap):
    totalCharge = 0
    for mcpeList in mcpeMap.values():
        charge, _time = getHitInformation(mcpeList)
        totalCharge += charge
    
    return totalCharge / len(mcpeMap)

def distanceCorrectionFactor(d_gamma):
    return np.sqrt(d_gamma**2 + d_1**2)

def calculateInitialGuess(frame, domsUsed, t_0):
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

    return q.x, q.y, q.z, u.z, phi, linefit

def get_t_0(frame):
    mcpeMap = frame["MCPESeriesMap_significant_hits"]
    mcpeList = mcpeMap[mcpeMap.keys()[0]]

    return mcpeList[0].time

def calculateVertex(q, u, t_0):
    delta_q = dataclasses.I3Position(u.x*c*t_0, u.y*c*t_0, u.z*c*t_0)

    return q - delta_q

# wrapper implemented so that input data can be passed and left untouched by minimizer
class QualityFunctor():

    def __init__(self, frame, t_0, printout = False):
        self.frame = frame
        self.printout = printout
        self.t_0 = t_0
    
    def qualityFunction(self, q_x, q_y, q_z, u_z, phi):
        q = dataclasses.I3Position(q_x, q_y, q_z)
        theta = np.arccos(u_z)
        rad_phi = np.radians(phi)
        u = dataclasses.I3Direction(np.sin(theta)*np.cos(rad_phi), np.sin(theta)*np.sin(rad_phi), u_z)

        mcpeMap = self.frame["MCPESeriesMap_significant_hits"]
        geoMap = geometry.omgeo
        averageCharge = getAverageCharge(mcpeMap)
        QSum = 0
        for omkey, mcpeList in mcpeMap:
            omPosition = geoMap[omkey].position

            # calculate expected values
            z_c = get_z_c(q, u, omPosition.x, omPosition.y)
            t_c = get_t_c(q, u, omPosition.x, omPosition.y, z_c, self.t_0)
            d_c = get_d_c(q, u, t_c, omPosition.x, omPosition.y, self.t_0)
            d_gamma = get_d_gamma(u, d_c, z_c, omPosition.z)
            t_gamma = get_t_gamma(u, t_c, d_gamma, z_c, omPosition.z)
            cos_theta_gamma = get_cos_theta_gamma(u, d_gamma, z_c, omPosition.z)

        # get data for om hits
            charge, t_i = getHitInformation(mcpeList)

        # calculate quality function components
            timePortion = (t_gamma - t_i)**2 / (sigma_i**2)
            A = chargeCorrectionFunction(charge, cos_theta_gamma)
            D = distanceCorrectionFactor(d_gamma)
            chargePortion = (A*D) / (averageCharge*d_0)

            # add to quality function sum
            QSum += timePortion + chargePortion
            
            if self.printout:
                printOutFile.write('measured: ' + str(t_i) + ', calculated: ' + str(t_gamma) + ', photon distance: ' + str(d_gamma) + ', charge portion: ' + str(chargePortion)
                                    + ', distance factor: ' + str(D) + ', charge factor: ' + str(A) + ', average charge: ' + str(averageCharge) + '\n' )

        if self.printout:
            printOutFile.write( "Qvalue: " + str(QSum) + '\n\n')
        
        return QSum

    def __call__(self, q_x, q_y, q_z, u_z, phi):
        return self.qualityFunction(q_x, q_y, q_z, u_z, phi)

# Main body for the improved track reco. 
def main():
    for infile in infileList:
        for frame in infile:
            if SimAnalysis.passFrame(frame, domsUsed, hitThresh, domThresh, maxResidual, geometry.omgeo):
                frame = SimAnalysis.writeSigHitsMapToFrame(frame, domsUsed, hitThresh, domThresh, maxResidual, geometry.omgeo)
                t_0 = get_t_0(frame)
                initialGuess = calculateInitialGuess(frame, domsUsed, t_0)

                qFunctor = QualityFunctor(frame, t_0)
                initialGuessQ = qFunctor(initialGuess[0], initialGuess[1], initialGuess[2], initialGuess[3], 
                                         initialGuess[4])
            
                # record seed values for comparison
                printOutFile.write("initial guess: " + str(initialGuess) + ", \nvalue of Q: " + str(initialGuessQ) + '\n')
                
                minimizer = Minuit(qFunctor, q_x = initialGuess[0], q_y = initialGuess[1],q_z = initialGuess[2], 
                                   u_z = initialGuess[3], phi = initialGuess[4], error_q_x = 1, error_q_y = 1, 
                                   error_q_z = 1, error_u_z = 0.05, error_phi = 1, errordef = 1, 
                                   limit_u_z = (-1, 1), limit_phi = (0, 360) )
                
                minimizer.migrad()

                # record minimizer results in output file
                printOutFile.write("results:\n")
                printOutFile.write(str(minimizer.get_fmin()) + '\n')
                printOutFile.write( str(minimizer.values) + '\n')
                printOutFile.write('\n\n')

                solution = minimizer.values
                q = dataclasses.I3Position(solution['q_x'], solution['q_y'], solution['q_z'])
                phi = solution['phi'] * I3Units.deg
                theta = np.arccos(solution['u_z'])
                u = dataclasses.I3Direction(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), solution['u_z'])

                # record final q function breakdown in output file
                printQFunctor = QualityFunctor(frame, t_0, printout = True)
                printQFunctor(solution['q_x'], solution['q_y'], solution['q_z'], solution["u_z"], solution["phi"])

                recoParticle = dataclasses.I3Particle()
                recoParticle.shape = dataclasses.I3Particle.InfiniteTrack
            
                # record on particle whether reconstruction was successful
                if minimizer.get_fmin()["is_valid"]:
                    recoParticle.fit_status = dataclasses.I3Particle.OK
                else:
                    recoParticle.fit_status = dataclasses.I3Particle.InsufficientQuality
                    
                recoParticle.dir = u
                recoParticle.speed = c
                recoParticle.pos = calculateVertex(q, u, t_0)
                recoParticle.time = 0
        
                # include both linefit and improved recos for comparison
                frame.Put('ImprovedRecoParticle', recoParticle)
                frame.Put('LineFitRecoParticle', initialGuess[5])
                outfile.push(frame)
                    
    outfile.close()
    printOutFile.close()


# Only runs the improved track reconstruction as written here if it is run as the main file. 
if __name__ == "__main__":
    main()
