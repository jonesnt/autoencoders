'''
This is an abstract class for image generation.
Image generator classes inherit from this class and implement the generate function.
'''
from abc import ABC, abstractmethod
from torch import Tensor

class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, width=64, height=64) -> Tensor:
        pass
