#!/usr/bin/env python

from icecube import dataclasses, dataio, icetray, simclasses
from icecube.icetray import I3Units, I3Frame, OMKey
import numpy as np
from simAnalysis import SimAnalysis
from gcdScripts import gcdHelpers
import argparse
from itertools import combinations

'''
A script that looks at the number of frames retained by different decompositions of
a dense geometry. This allows to quickly discard geometries that are not of sufficient
quality without having to run a full simulation set for them. Creates new Geometries for 
ones that passed and a new I3File with the events that passed. 
'''

parser = argparse.ArgumentParser(description = "Creates a reconstruction of the muon track using a linear least squares fit on the pulses")
#parser.add_argument( '-n', '--minFileNum', help = "smallest file number used" )
#parser.add_argument( '-N', '--maxFileNum', help = "largest file number used")
parser.add_argument( '-H', '--hitThresh', help = "threshold of hits for the DOM to be considered")
parser.add_argument( '-D', '--domThresh', help = "threshold of hit DOMs for the frame to be considered")
parser.add_argument( '-R', '--maxResidual', default = 100 , help = "maximum time residual allowed for the hit to be considered")
#parser.add_argument( '-l', '--numLayers', help = "number of layers the decomposed geometry will have")
parser.add_argument( '-o', '--outputName', help = "output name chosen for this specific iteration of the geometry")
args = parser.parse_args()

hitThresh = int(args.hitThresh)
domThresh = int(args.domThresh)
maxResidual = float(args.maxResidual)

infileList = []
for i in range(500, 700):
    infile = dataio.I3File('/home/dvir/workFolder/I3Files/nugen/nugenStep3/denseGeo/NuGen_step3_denseGeo_' + str(i) + '.i3.gz')
    infileList.append(infile)

gcdfile = dataio.I3File('/home/dvir/workFolder/I3Files/gcd/denseGeo/denseGeo_n30_b50.0_a4.5_l7_linear_reset_offset_simple_spacing.i3.gz')
densegeometry = gcdfile.pop_frame()["I3Geometry"]

# easier to just process one DOM line at a time
comparisonStringList = [i*2+0*30 for i in range(1,16)]
comparisonStringList.extend([i*2+4*30 for i in range(2,16)])
comparisonStringList.extend([i*2+8*30 for i in range(3,16)])
comparisonStringList.extend([i*2+12*30 for i in range(2,16)])
comparisonStringList.extend([i*2+16*30 for i in range(3,16)])
layerInfo = [1,3,5]

# spacing of 100m limits possible combinations

def convertToOmkeyList(stringList, layers):
    omkeyList = []
    for string in stringList:
        for dom in layers:
            omkeyList.append(OMKey(string, dom, 0))

    return omkeyList
 
omkeys = convertToOmkeyList(comparisonStringList, layerInfo)
retainedFramesComp = SimAnalysis.calculateRetainedFrames(infileList, omkeys, hitThresh, domThresh, maxResidual, densegeometry.omgeo)
print retainedFramesComp, len(omkeys)

testList = [i*2+0*30 for i in range(1,12)]
#testList.extend([i*2+1*30 for i in range(2,10)])
testList.extend([i*2+2*30 for i in range(8,12)])
#testList.extend([i*2+3*30 for i in range(2,10)])
testList.extend([i*2+4*30 for i in range(2,12)])
#testList.extend([i*2+5*30 for i in range(2,10)])
testList.extend([i*2+6*30 for i in range(8,12)])
#testList.extend([i*2+7*30 for i in range(3,10)])
testList.extend([i*2+8*30 for i in range(3,12)])
#testList.extend([i*2+9*30 for i in range(2,10)])
testList.extend([i*2+10*30 for i in range(8,12)])
#testList.extend([i*2+11*30 for i in range(2,10)])
testList.extend([i*2+12*30 for i in range(2,12)])
#testList.extend([i*2+13*30 for i in range(2,10)])
testList.extend([i*2+14*30 for i in range(8,12)])
#testList.extend([i*2+15*30 for i in range(3,10)])
testList.extend([i*2+16*30 for i in range(3,12)])
#testList.extend([i*2+17*30 for i in range(3,10)])
testList.extend([i*2+18*30 for i in range(8,12)])
#testList.extend([i*2+19*30 for i in range(3,10)])

layerInfo = [1,3,5]

omkeys = convertToOmkeyList(testList, layerInfo)
#retainedFramesTest = SimAnalysis.calculateRetainedFrames(infileList, omkeys, hitThresh, domThresh, maxResidual, densegeometry.omgeo)
#print retainedFramesTest, len(omkeys)

if retainedFramesTest > 3000:
    print "passed"
    outfile = dataio.I3File('/home/dvir/workFolder/I3Files/nugen/nugenStep3/partialDenseGeo/NuGen_step3_partialDenseGeo_' + str(args.outputName) + '.i3.gz', 'w')
    gcdOutfile = dataio.I3File('/home/dvir/workFolder/I3Files/gcd/partialDenseGeo/partialDenseGeo_' + str(args.outputName) + '.i3.gz', 'w')
    
    newGeometry = gcdHelpers.makePartialGeometry(densegeometry, omkeys)

    # generate new frames
    gframe = icetray.I3Frame(icetray.I3Frame.Geometry) 
    cframe = gcdHelpers.generateCFrame(newGeometry)
    dframe = gcdHelpers.generateDFrame(newGeometry)

    # add keys and values to G frame
    gframe["I3Geometry"] = newGeometry

    # push frames onto output file
    gcdOutfile.push(gframe)
    gcdOutfile.push(cframe)
    gcdOutfile.push(dframe)

    # close output file
    gcdOutfile.close()

    for infile in infileList:
        infile.rewind()
        for frame in infile:
            if SimAnalysis.passFrame(frame, omkeys, hitThresh, domThresh, maxResidual, densegeometry.omgeo):
                outfile.push(frame)

    outfile.close()