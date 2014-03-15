import numpy as np
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as color

import glob

from getEigenface import keyNpz

SAMPLE_IMAGE_NAME='yalefaces/subject13.wink.gif'
SAMPLE_MEAN = 'meanYale.npz'

# Composes new images too
class Recognizer:
    def __init__(self, eigfaceFile, k, sampleImage=SAMPLE_IMAGE_NAME, 
                 meanNpz=SAMPLE_MEAN):

        self.imsize = misc.imread(SAMPLE_IMAGE_NAME).shape

        self.PCs = [] # principal components
        npzfile = np.load(eigfaceFile, mmap_mode=None)
        eigfs = sorted(npzfile.files, key=keyNpz)
        if k >= len(eigfs):
            raise ValueError("k more than number of PC's")

        self.eigfaces = []
        for i in xrange(len(eigfs)):
            ef = npzfile[eigfs[i]]
            ef /= np.linalg.norm(ef)
            self.eigfaces.append(ef)

        # we got eigenfaces from smallest to largest eigenvalue
        self.eigfaces = self.eigfaces[::-1]

        for i in xrange(k):
            self.PCs.append(self.eigfaces[i])

        npzfile = np.load(meanNpz, mmap_mode=None)
        self.meanVec = npzfile[npzfile.files[0]]

    def showPCs(self, save=False):
        i = 0
        for pc in self.PCs:
            i += 1
            eigface = np.float32(pc)
            im = np.reshape(eigface, self.imsize)
            if save:
                picName = 'pc_' + str(i) + '.png'
                plt.imsave(picName, im, cmap=pylab.gray())
                
            plt.figure(i)
            plt.imshow(im, cmap=pylab.gray())


    def represent(self, targetFilename):
        print "represent!"
        targetIm = misc.imread(targetFilename, flatten=True)
        targetIm = targetIm# / 255.0

        faceVec = np.asarray(targetIm).reshape(-1)
        # project onto PC to get coefficients (coordinates)

        faceVec = faceVec / np.float32(255)

        print faceVec
        print "AAAAAAAAAAAAAAAAAAAAAAA"
        print self.meanVec
        meanIm = np.reshape(self.meanVec, self.imsize)
        plt.imshow(meanIm, cmap=pylab.gray())
        plt.show()

        diffVec = faceVec - self.meanVec
        print "DDDDD DD DD DDDDD"
        print diffVec

        numPCs = len(self.PCs)
        coords = []
        for pc in self.PCs:
            coord = diffVec.dot(pc)
            coords.append(coord)

        limit = max(coords)
        print "BBBBBBBBBBBBBBBBBBB"
        print coords
        print "PCPCPCPCPCPCPCPCPC"
        print self.PCs

        repVec = coords[0] * self.PCs[0]
        for i in xrange(1, numPCs):
            pc = self.PCs[i]
            coord = coords[i]
            a = (coord * pc)
            print "Brin", i
            print a
            repVec += a
            #repIm = np.reshape(repVec, self.imsize)
            #plt.imshow(repIm, cmap=pylab.gray())
            #plt.show()
            print repVec

        repVec += self.meanVec

        repIm = np.reshape(repVec, self.imsize)


        print "VANDAM"
        print repIm

        repIm = repIm * np.float32(255)

        print "BLUELUELUE"
        print repIm

        rows = repIm.shape[0]
        cols = repIm.shape[1]
        M255 = np.float32(255)
        for r in xrange(rows):
            for c in xrange(cols):
                if repIm[r][c] > M255:
                    repIm[r][c] = M255
                if repIm[r][c] < 0:
                    repIm[r][c] = 0

        print "NAUTILUS"
        print repIm/M255
        repIm = repIm/M255

        plt.imshow(repIm, cmap=pylab.gray())
        plt.show()
        #        '''

if __name__ == '__main__':
    r = Recognizer('eig3.npz', 80)
    #r.showPCs()
    r.represent('./yalefaces/subject01.normal.gif')

