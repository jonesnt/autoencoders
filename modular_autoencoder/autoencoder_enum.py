from enum import Enum
from autoencoders.basic_autoencoder import ImageAutoencoder
from autoencoders.test_autoencoder import TestAutoencoder

class AUTOENCODERS(Enum):
    BASIC = ImageAutoencoder
    TEST = TestAutoencoder
