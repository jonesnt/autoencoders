import torch
import math
import numpy as np

from simple_generator import SimpleGenerator
from ortho import ortho

class BodyCenteredGenerator(SimpleGenerator):

    def generate(self, width=64, height=64):
        
        super().generate(width, height)
        
        simple_image = ortho()
        
        body_centered_image = torch.zeros((width, height), dtype=torch.float32)
        
        for u in range(width):
            Ku = u + 0.5  # Offset by 0.5 to get values from 0.5 to 63.5
            for v in range(height):
                Kv = v + 0.5  # Offset by 0.5 to get values from 0.5 to 63.5

                w = (simple_image['d'] - simple_image['alpha'] * simple_image['n1'] * Ku
                     - simple_image['beta'] * simple_image['n2'] * Kv) / simple_image['n3']
                Kw = np.floor((w/simple_image['gamma'])-0.5) + 0.5

                pointOnPlane = (
                    torch.tensor([Ku, Kv, Kw], dtype=torch.float32)
                    + simple_image['n3'] * (w / simple_image['gamma'] - Kw)
                    * torch.tensor([simple_image['n1'], simple_image['n2'], simple_image['n3']], dtype=torch.float32)
                )

                atomCoordinates = torch.matmul(simple_image['matrix_A'], pointOnPlane)