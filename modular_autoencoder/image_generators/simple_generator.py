import torch
import math
import numpy as np

from image_generator import ImageGenerator
from ortho import ortho

class SimpleGenerator(ImageGenerator):
    def generate(self, width=64, height=64):
        image_dat = ortho()
        image = torch.zeros((width, height), dtype=torch.float32)


        for Ku in range(width/image_dat['alpha']):
            for Kv in range(height/image_dat['beta']):
                # height on the plane
                w = (image_dat['d'] + image_dat['alpha'] * image_dat['n1'] * Ku
                    + image_dat['beta'] * image_dat['n2'] * Kv)/(image_dat['n3'])
                Kw = np.floor(w/image_dat['gamma'])

                # Row vector
                pointOnPlane = torch.tensor([Ku,Kv,Kw], dtype=torch.float32) + 
                                n3*(w/image['gamma']-Kw)*
                            torch.tensor([image_dat['n1'], image_dat['n2'], image_dat['n3']])
                

                AtomCoordinates = image_dat['matrix_A']*pointOnPlane





                





