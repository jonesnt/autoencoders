'''
This is a singleton handler class for images.
It accepts instructions about the number of images to generate and their specifications.
It returns a list of pseudorandom images for the purpose of model training.
'''

class Image_Handler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

