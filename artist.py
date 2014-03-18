import numpy as np
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as color

import glob

from getEigenface import keyNpz
from recognizer import Recognizer

M255 = np.float32(255)
K = 130
THRESHOLD_CLASS_FACE = 1000.

bor = True
loadPrior = True
#loadPrior = False
if bor:
    TEST_NAME = 's13_wink'
    SAMPLE_IMAGE_NAME='yalefaces/subject13.wink.gif'
    SAMPLE_MEAN = 'meanY.npz'
    if loadPrior:
        EIG_x_NPZ = 'normalized_eigY.npz'
    else:
        EIG_x_NPZ = 'eigY.npz'
else:
    TEST_NAME = 'face1a'
    SAMPLE_IMAGE_NAME='faces/1a.jpg'
    SAMPLE_MEAN = 'mean2.npz'
    EIG_x_NPZ = 'eigV2.npz'

class Artist:
    # <target> is from <folder>
    def __init__(self):
        # load resources via Recognizer
        self.recognizer = Recognizer(EIG_x_NPZ, K, loadPrior=True)

    def caricature(self, target, knownIndexes=[]):
        # enlarge top 3 coords
        # decrease bot 3 coords?
        targetCoords = self.recognizer.represent(target, makeFaceClass=True)

        # find top 3 coords
        idx1, coord1 = -1, 0
        idx2, coord2 = -1, 0
        idx3, coord3 = -1, 0
        numCoords = len(targetCoords)
        for i in xrange(numCoords):
            coord = targetCoords[i]
            if coord > coord1:
                # shift (1st, 2nd) to (2nd, 3rd)
                coord3 = coord2
                idx3 = idx2
                coord2 = coord1
                idx2 = idx1
                # new biggest coord
                coord1 = coord
                idx1 = i
            elif coord > coord2:
                # shift (2nd) to (3rd)
                coord3 = coord2
                idx3 = idx2
                # new 2nd biggest coord
                coord2 = coord
                idx2 = i
            elif coord > coord3:
                # new 3rd biggest coord
                coord3 = coord
                idx3 = i
        
        # magnify top 3 coords
        targetCoords[idx1] *= 2.
        targetCoords[idx2] *= 2.
        targetCoords[idx3] *= 2.
        

        self.recognizer.represent(target, coords=targetCoords)

        # (1) decide which eig facces to use?
        # OR (2( make caricature


if __name__ == '__main__':
    a = Artist()
    print "Finished initializing...\n"
    a.caricature(SAMPLE_IMAGE_NAME)
