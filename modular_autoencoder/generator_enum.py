from enum import Enum
from image_generators.simple_generator import SimpleGenerator
from image_generators.body_centered_generator import BodyCenteredGenerator

class GENERATORS(Enum):
   SIMPLE = SimpleGenerator
   BODY_CENTERED = BodyCenteredGenerator
