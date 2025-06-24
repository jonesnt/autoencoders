'''
This is a singleton handler class for images.
It accepts instructions about the number of images to generate and their specifications.
It returns a list of pseudorandom images for the purpose of model training.
'''
from image_generator import ImageGenerator
from generator_enum import GENERATORS
import random

class ImageHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def generate_training_data(self, num_images=100, width=64, height=64):
        data = []
        for i in range(num_images):
            # Randomly select a generator from the enum
            selected_generator_enum = random.choice(list(GENERATORS))
            # Get the class from the enum
            generator_class = selected_generator_enum.value
            # Create an instance of the generator
            generator_instance = generator_class()
            # Generate the image
            image = generator_instance.generate(width, height)
            data.append(image)
        return data
