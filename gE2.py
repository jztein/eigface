# 
import numpy as np
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as color

import glob

# because the loaded arrays are returned in a numpy
# dict object, keys called arr_1, arr_2, ...
def keyNpz(arr_x):
    return int(arr_x[4:])

SAMPLE = '/1a.jpg'

class Eigenfacer:
    def __init__(self, vecFaceFile, eigFaceFile, folder='', extension='', iAmNew=False):
        self.vecfaceFile = vecFaceFile
        self.eigfaceFile = eigFaceFile
        self.vecfaces = []
        self.mean = None
        self.npzfile = None
        self.mat = None
        self.cov = None
        self.eigenfaces = []

        img = misc.imread(folder + SAMPLE)
        self.imsize = img.shape

        if iAmNew:
            print 'Making vectorized faces'
            files = glob.glob(folder + '/*' + extension)
            self.makeFaceVectors(files)
        else:
            print 'Loading face vectors'
            # we have previously saved vectorized faces
            self.npzfile = np.load(self.vecfaceFile, mmap_mode=None)
            npzFiles = sorted(self.npzfile.files, key=keyNpz)
            for v in npzFiles:
                self.vecfaces.append(self.npzfile[v] / np.float32(255))

    def __del__(self):
        pass

    def makeFaceVectors(self, files):
        # vectorize all images
        for f in files:
            img = misc.imread(f)
            self.vecfaces.append(np.asarray(img).reshape(-1))
        np.savez_compressed(self.vecfaceFile, *(self.vecfaces))

    def printFaceVector(self, idx=0):
        print self.vecfaces[idx]

    def getMean(self):
        # mean face vector
        sumArr = np.zeros(self.vecfaces[0].size)
        numVecs = len(self.vecfaces)
        for v in self.vecfaces:
            sumArr += v
        self.mean = sumArr / numVecs
        self.mean = np.float32(self.mean)
        print 'MEAN:', self.mean

        imMean = np.reshape(self.mean, self.imsize)

        print '****', self.imsize

        #print 'IMEAN'
        #print imMean

        plt.imsave('meanImage.png', imMean, cmap=pylab.gray())
        np.savez_compressed("mean2.npz", self.mean)
        #plt.show()
        
    def centerFaceVectors(self):
        if self.mean == None:
            raise ValueError('Cannot centerFaceVectors(), mean not calculated')

        print 'Center data around origin (X_i - MeanX)'
        print 'Old first vector:'
        self.printFaceVector()
        
        numVecs = len(self.vecfaces)

        self.mat = np.zeros((numVecs, self.vecfaces[0].size))

        print "##########"
        print self.mean
        print "##########"
        print self.vecfaces[0]
        print "##########"

        for i in xrange(numVecs):
            self.vecfaces[i] = self.vecfaces[i] - self.mean
            self.mat[i,:] = self.vecfaces[i]# - self.mean

        print 'New first vector:'
        self.printFaceVector()

        print 'Check mat 1st vector entry'
        print self.mat[0, :]

    def getEigenfaces(self):
        # get covariance matrix of all face vectors
        # centralized around origin
        self.getMean()

        

        self.centerFaceVectors()
        A_At = self.mat.dot(self.mat.T) # matrix multiplication
        print "finished mat mult"
        self.cov = np.cov(A_At)

        eigVals, eigVecs = np.linalg.eigh(self.cov)

        print "#########################################"
        print "#########################################"
        print eigVals
        print "#########################################"
        print eigVecs
        print "#########################################"
        print "#########################################"

        print eigVals.shape, self.mat.T.shape
        print "========"
        print eigVecs.shape

        #Get eigenfaces from eigenVecs
        eigFaces = self.mat.T.dot(eigVecs)
        print "SHAPE", eigFaces.shape
        numPixels = eigFaces.shape[0]
        numEigenfaces = eigFaces.shape[1]

        for i in xrange(numEigenfaces):
            if i < 10:
                continue
            eigFace = eigFaces[:,i]
            '''
            for i in xrange(numPixels):
                if eigFace[i] > 1:
                    eigFace[i] = 1
                elif eigFace[i] < 0:
                    eigFace[i] = 0
                    '''
            self.eigenfaces.append(eigFace)

        self.eigenfaces = self.eigenfaces[::-1]

        print "Finished getting eigenfaces..."
        np.savez_compressed(self.eigfaceFile, *(self.eigenfaces))

    def showEigFace(self, idx=0):
        eigface = np.float32(self.eigenfaces[idx])
        #print 'Eigface', eigface
        #print 'SHAPE:', eigface.shape
        im = np.reshape(eigface, self.imsize)
        plt.imsave('eigFaceImage'+str(i)+'.png', im, cmap=pylab.gray())

    def readEigFaces(self):
        self.npzfile = np.load(self.eigfaceFile, mmap_mode=None)
        npzFiles = sorted(self.npzfile.files, key=keyNpz)
        for v in npzFiles:
            self.eigenfaces.append(self.npzfile[v])

if __name__ == '__main__':
    #e = Eigenfacer('vec1.npz', 'eig1.npz', './faces', '.jpg')
    e = Eigenfacer('vecV2.npz', 'eigV2.npz', './faces', '.jpg', iAmNew=True)
    #e.printFaceVector()

    #e = Eigenfacer('vecTest2.npz', 'eigTest2.npz', './yalefaces', '.gif', iAmNew=False)#True)

    e.getEigenfaces()

    #e.readEigFaces()
    #'''
    for i in xrange(100):
        e.showEigFace(i)
    #'''
