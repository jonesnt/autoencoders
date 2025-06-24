'''
This is an abstract class for image generation.
Image generator classes inherit from this class and implement the generate function.
'''
from abc import abstractmethod

class ImageGenerator:
    @abstractmethod
    def generate(self, width=64, height=64):
        pass
