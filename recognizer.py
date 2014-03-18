import numpy as np
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as color

import glob

from getEigenface import keyNpz

M255 = np.float32(255)
K = 130
THRESHOLD_CLASS_FACE = 0.1

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


# Composes new images too
class Recognizer:
    def __init__(self, eigfaceFile, k, sampleImage=SAMPLE_IMAGE_NAME, 
                 meanNpz=SAMPLE_MEAN, loadPrior=False):

        self.k = k
        self.classFaces = []

        img = misc.imread(SAMPLE_IMAGE_NAME)
        self.imsize = img.shape
        #self.imsize = misc.imread(SAMPLE_IMAGE_NAME).shape

        self.PCs = [] # principal components
        npzfile = np.load(eigfaceFile, mmap_mode=None)
        eigfs = sorted(npzfile.files, key=keyNpz)
        #print npzfile.files
        if k >= len(eigfs):
            raise ValueError("k more than number of PC's")

        self.eigfaces = []
        for i in xrange(len(eigfs)):
            ef = npzfile[eigfs[i]]
            #ef = np.float32(ef)
            if not loadPrior:
                ef /= np.linalg.norm(ef)
            self.eigfaces.append(ef)

        # we got eigenfaces from smallest to largest eigenvalue
        #self.eigfaces = self.eigfaces[::-1]

        for i in xrange(k):
            self.PCs.append(self.eigfaces[i])

        if not loadPrior:
            np.savez_compressed('normalized_' + eigfaceFile, *(self.eigfaces))

        npzfile = np.load(meanNpz, mmap_mode=None)
        self.meanVec = npzfile[npzfile.files[0]]

    def showPCs(self, save=False):
        i = 0
        for pc in self.PCs:
            i += 1
            eigface = pc
            eigface = np.float32(pc)
            im = np.reshape(eigface, self.imsize)
            if save:
                picName = 'pc_' + str(i) + '.png'
                plt.imsave(picName, im, cmap=pylab.gray())

            plt.figure(i)
            plt.imshow(im, cmap=pylab.gray())


    def represent(self, targetFilename, makeFaceClass=False, coords=[]):
        print "represent!"
        targetIm = misc.imread(targetFilename)
        plt.imshow(targetIm, cmap=pylab.gray())
        plt.figure(2)
        faceVec = np.asarray(targetIm).reshape(-1)
        # project onto PC to get coefficients (coordinates)

        faceVec = np.float32(faceVec)

        diffVec = faceVec - self.meanVec

        #diffVec /= np.float32(255)

        diffIm = np.reshape(diffVec, self.imsize)
        #plt.imshow(diffIm, cmap=pylab.gray())
        #plt.show()
        #exit(1)
        #plt.figure(2)

        numPCs = len(self.PCs)
        if not coords:
            coords = []
            for pc in self.PCs:
                coord = diffVec.dot(pc)
                coords.append(coord)

            if makeFaceClass:
                return coords

        print "BBBBBBBBBBBBBBBBBBB"
        print coords
        print "PCPCPCPCPCPCPCPCPC"
        print self.PCs


        repVec = self.meanVec[:]

        repVec += coords[0] * self.PCs[0]
        repIm = np.reshape(repVec, self.imsize)
        plt.subplot(13, 10, 1)
        plt.imshow(repIm, cmap=pylab.gray())
        for i in xrange(1, numPCs):
            pc = self.PCs[i]
            coord = coords[i]
            a = (coord * pc)

            #print "Brin", i
            #print a
            repVec += a
            repIm = np.reshape(repVec, self.imsize)
            plt.imsave('brin' + str(i), repIm, cmap=pylab.gray())
            #plt.subplot(13, 10, i)
            #plt.imshow(repIm, cmap=pylab.gray())
            #plt.setp(repIm, 'XTickLabel', [],'XTick',[])
            #plt.show()
            #print repVec

        #plt.show()

        repIm = np.reshape(repVec, self.imsize)
        plt.imshow(repIm, cmap=pylab.gray())

        plt.figure(3)
        meaner = np.reshape(self.meanVec, self.imsize)
        plt.imshow(meaner, cmap=pylab.gray())
        print meaner
        print "VANDAM"
        print repIm
        print "STUFFER"
        print diffIm

        rows = repIm.shape[0]
        cols = repIm.shape[1]
        count = 0
        whites = 0
        blacks = 0
        for r in xrange(rows):
            for c in xrange(cols):
                count += 1

                if repIm[r][c] > M255:
                    repIm[r][c] = M255
                    whites += 1

                if repIm[r][c] < 0:
                    repIm[r][c] = 0
                    blacks += 1

        print whites / float(count), blacks / float(count)

        plt.imshow(repIm, cmap=pylab.gray())
        plt.show()
        plt.imsave(TEST_NAME + '_f1_' + str(K), repIm, cmap=pylab.gray())
        #        '''

    # recognize
    def recognize(self, target, folder):
        targetDiffVec = self.represent(target, makeFaceClass=True)
        vecLen = len(targetDiffVec)

        self.classFaces = [] # clear

        files = glob.glob(folder + '/*.gif')
        minEuclidean = -1
        closest = 0
        count = 0
        for f in files:
            faceClass = self.represent(f, makeFaceClass=True)
            self.classFaces.append(faceClass)
            curEuc = 0
            for i in xrange(vecLen):
                curEuc += (faceClass[i]-targetDiffVec[i])**2
            curEuc = abs(curEuc)
            if minEuclidean > curEuc or minEuclidean < 0:
                minEuclidean = curEuc
                closest = count
            count += 1

        if minEuclidean < 0:
            raise ValueError("SOMETHOING HEAOGNOUSHELY WRONG")

        if minEuclidean < THRESHOLD_CLASS_FACE:
            print 'Closest to face %d' % closest
            self.represent(target, coords=self.classFaces[closest])
            
        else:
            print "<<<<<< No match found"

if __name__ == '__main__':
    r = Recognizer(EIG_x_NPZ, K, loadPrior=True)
    #r = Recognizer(EIG_x_NPZ, K, loadPrior=False)

    r.recognize(SAMPLE_IMAGE_NAME, 'yalefaces')
    exit(1)

    show = False#True
    if show:
        r.showPCs()
        plt.show()

    r.represent(SAMPLE_IMAGE_NAME)


