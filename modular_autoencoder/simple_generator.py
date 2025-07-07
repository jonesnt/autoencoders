import torch
import math
import numpy as np
import scipy

from image_generator import ImageGenerator
from ortho import ortho

class SimpleGenerator(ImageGenerator):
    def generate(self, width=64, height=64):
        image_dat = ortho(spread=10)
        # top left is (0,0), first dimension is x, second y
        # gray scale 0-255
        image = torch.zeros((width, height), dtype=torch.float32)


        # for (u,v) pair add atom blur to plane
        for Ku in range(int(np.floor(width/image_dat['alpha']))):
            for Kv in range(int(np.floor(height/image_dat['beta']))):
                # height on the plane
                w = (image_dat['d'] + image_dat['alpha'] * image_dat['n1'] * Ku
                    + image_dat['beta'] * image_dat['n2'] * Kv)/(image_dat['n3'])
                Kw = np.floor(w/image_dat['gamma'])

                # Row vector
                pointOnPlane = torch.tensor([Ku,Kv,Kw], dtype=torch.float32) + image_dat['n3']*(w/image_dat['gamma']-Kw)*torch.tensor([image_dat['n1'], image_dat['n2'], image_dat['n3']])

                AtomCoordinates = image_dat['matrix_A']*pointOnPlane
                
                # pytorch should multiply the matrix with vector even though the dimensions dont match
                atomCoordinates = torch.matmul(image_dat['matrix_A'], pointOnPlane)

                #TODO: Define standDev and constant, should depend on spread, distance, and atom size
                constant = 1
                standDev = 1
                atomWidth = np.log(np.exp(1/(2*standDev**2)/(1000*constant)))
                for pixelX in range(int(np.floor(2*atomWidth))):
                    for pixelY in range(int(np.floor(2*atomWidth))):
                        # if not outside of the frame
                        if(atomCoordinates[0] - atomWidth >= 0 and atomCoordinates[1] - atomWidth >= 0):
                            # Calculates Gaussian Blur of each atom
                            f = lambda y, x: constant*scipy.exp((-(atomCoordinates[0]-x)**2 -(atomCoordinates[1]-7)**2)/(2*standDev**2))
                            image[pixelX][pixelY] = scipy.integrate.dblquad(f, pixelX, pixelX+1, pixelY, pixelY+1)

        return image





                





