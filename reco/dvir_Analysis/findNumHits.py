#!/usr/bin/env python

from icecube import dataclasses, dataio, simclasses
from icecube.icetray import I3Units, OMKey, I3Frame
from icecube.dataclasses import ModuleKey
import matplotlib.pyplot as plt
import numpy as np

step3infile = dataio.I3File('/home/dvir/workFolder/I3Files/muongun/muongun_step3/MuonGun_step3_139005_000000.i3.bz2')
custominfile = dataio.I3File('/home/dvir/workFolder/I3Files/muongun/customGenHitsMuongun/MuonGun_customGenHits_0.i3.gz')
gcdFile = dataio.I3File('/home/dvir/workFolder/I3Files/gcd/IceCube/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz')
geometry = gcdFile.pop_frame(I3Frame.Geometry)["I3Geometry"]

# get all Q frames
step3qframes = []
while(step3infile.more()):
    step3qframes.append(step3infile.pop_daq())

customqframes = []
while(custominfile.more()):
    customqframes.append(custominfile.pop_daq())

largeRatioFrames = []
smallRatioFrames = []
largeRatioFramesCust = []
smallRatioFramesCust = []

# create a dictionary of event header and frame to ensure events match up
evtFrameMap = {}
for frame in customqframes:
    header = frame["I3EventHeader"]
    evtFrameMap[header.event_id] = frame
    
def getNumHitsInFrame(frame):
    mcpeSeriesMap = frame["MCPESeriesMap"]
    numHits = 0

    for omkey in mcpeSeriesMap.keys():
        hits = mcpeSeriesMap[omkey]
        numHits += len(hits)
    
    return numHits

def getrelativeAngle(photon, omkey):
    geoMap = geometry.omgeo
    domGeo = geoMap[omkey]
    domDirection = domGeo.direction
    photonDirection = photon.dir
    dotProduct = photonDirection.x*domDirection.x + photonDirection.y*domDirection.y + photonDirection.z*domDirection.z

    return -dotProduct

def getPhotonDataFromFrames(frameList):
    photonAngle = []
    photonWavelength = []

    for frame in frameList:
        photonMap = frame["I3Photons"]
        for modkey, photons in photonMap:
            for photon in photons:
                photonAngle.append( getrelativeAngle( photon, OMKey(modkey.string, modkey.om, 0) ) )
                photonWavelength.append(photon.wavelength)

    
    return photonAngle, photonWavelength

def getDOMDataFromFrames(frameList):
    numDOMs = []

    for frame in frameList:
        mcpeMap = frame["MCPESeriesMap"]
        numDOMs.append(len(mcpeMap.keys()))
    
    return numDOMs


step3Hits = []
customHits = []
event_id = []

for frame in step3qframes:
    header = frame["I3EventHeader"]
    evt_id = header.event_id
    if evt_id in evtFrameMap.keys():
        event_id.append(evt_id)
        customFrame = evtFrameMap[evt_id]
        step3Hits.append(getNumHitsInFrame(frame))
        customHits.append(getNumHitsInFrame(customFrame))
        if step3Hits[len(step3Hits)-1]/float(customHits[len(customHits)-1]) > 2:
            largeRatioFrames.append(frame)
            largeRatioFramesCust.append(customFrame)
        elif step3Hits[len(step3Hits)-1]/float(customHits[len(customHits)-1]) < 0.5:
            smallRatioFrames.append(frame)
            smallRatioFramesCust.append(customFrame)

    else:
        hits = getNumHitsInFrame(frame)
# analysis of how different the frames are

print(len(largeRatioFrames))
print(len(smallRatioFrames))

ratio = []
for i in range(len(step3Hits)):
    ratio.append( step3Hits[i]/float(customHits[i]) )

plt.figure()    
plt.hist(ratio, bins = 35, histtype = 'step', range = (0,5))
plt.title("Ratio of Number of Hits Seen by Step 3 to Custom HitGen")
plt.xlabel("# step 3 hits / # custom hits")
plt.ylabel("# of occurences")
#plt.show()

# analyzing cause for large ratio frames
photonAngleLR, photonWavelengthLR = getPhotonDataFromFrames(largeRatioFrames)
photonAngleSR, photonWavelengthSR = getPhotonDataFromFrames(smallRatioFrames)
photonAngleTF, photonWavelengthTF = getPhotonDataFromFrames(step3qframes)
numDOMsLRstep3 = getDOMDataFromFrames(largeRatioFrames)
numDOMsTFstep3 = getDOMDataFromFrames(step3qframes)
numDOMsSRstep3 = getDOMDataFromFrames(smallRatioFrames)
numDOMsLRcust = getDOMDataFromFrames(largeRatioFramesCust)
numDOMsTFcust = getDOMDataFromFrames(customqframes)
numDOMsSRcust = getDOMDataFromFrames(smallRatioFramesCust)

'''
plt.figure()
plt.hist(photonAngleLR, bins = 20, histtype = 'step')
plt.title("Angular distribution of photons for large ratio frames")
plt.xlabel("Relative angle between DOM and photon (degrees)")
plt.ylabel("# of occurences")
#plt.show()

plt.figure()
plt.hist(photonWavelengthLR, bins = 20, histtype = 'step')
plt.title("Wavelength distribution fof photons for large ratio frames")
plt.xlabel("wawvelength (nm)")
plt.ylabel("# of occurences")
#plt.show()

#plt.figure()
#plt.hist(photonAngleSR, bins = 20, histtype = 'step')
#plt.title("Angular distribution of photons for small ratio frames")
#plt.xlabel("Relative angle between DOM and photon (degrees)")
#plt.ylabel("# of occurences")
#plt.show()

#plt.figure()
#plt.hist(photonWavelengthSR, bins = 20, histtype = 'step')
#plt.title("Wavelength distribution fof photons for small ratio frames")
#plt.xlabel("wawvelength (nm)")
#plt.ylabel("# of occurences")
#plt.show()

plt.figure()
plt.hist(photonAngleTF, bins = 20, histtype = 'step')
plt.title("Angular distribution of photons for all frames")
plt.xlabel("Relative angle between DOM and photon (degrees)")
plt.ylabel("# of occurences")
#plt.show()

plt.figure()
plt.hist(photonWavelengthTF, bins = 20, histtype = 'step')
plt.title("Wavelength distribution fof photons for all frames")
plt.xlabel("wawvelength (nm)")
plt.ylabel("# of occurences")
plt.show()
'''
'''
plt.figure()
plt.hist(numDOMsLRstep3, bins = 20, histtype = 'step', color = 'skyblue', label = "step3")
plt.hist(numDOMsLRcust, bins = 20, histtype = 'step', color = 'red', label = "custom")
plt.title("Distribution of the number of DOMs that saw light in LR frames")
plt.xlabel("Number of DOMs with hits")
plt.ylabel("# of occurences")
plt.legend()
#plt.show()

plt.figure()
plt.hist(numDOMsTFstep3, bins = 20, histtype = 'step', color = 'skyblue', label = "step3")
plt.hist(numDOMsTFcust, bins = 20, histtype = 'step', color = 'red', label = "custom")
plt.title("Distribution of the number of DOMs that saw light in all frames")
plt.xlabel("Number of DOMs with hits")
plt.ylabel("# of occurences")
plt.legend()
#plt.show()
'''

plt.figure()
plt.hist(numDOMsTFstep3, bins = 20, histtype = 'step', color = 'skyblue', label = "IceCube Package")
plt.hist(numDOMsTFcust, bins = 20, histtype = 'step', color = 'red', label = "Custom Script")
plt.title("Distribution of the number of DOMs that saw light")
plt.xlabel("Number of DOMs with hits")
plt.ylabel("# of occurences")
plt.legend()
#plt.show()


from scipy.stats import norm
import matplotlib.mlab as mlab

plt.figure()

# read data from a text file. One number per line
logRatio = np.log(ratio)

# best fit of data
(mu, sigma) = norm.fit(logRatio)

# the histogram of the data
n, bins, patches = plt.hist(logRatio, 60, normed=1, facecolor='green', alpha=0.75, label = "mu: " + str(mu) + "\nsigma: " + str(sigma))

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

#plot
plt.xlabel('log ratio')
plt.ylabel('Probability')
plt.legend()

#plt.show()

plt.figure()
plt.hist(step3Hits, bins = 20, histtype = 'step', color = 'skyblue', label = "IceCube Package, total hits: " + str(sum(step3Hits)))
plt.hist(customHits, bins = 20, histtype = 'step', color = 'red', label = "Custom Script, total hits: " + str(sum(customHits)))
plt.title("Amount of Hits Per Event")
plt.xlabel("Number of Hits")
plt.ylabel("# of occurences")
plt.legend()
plt.show()