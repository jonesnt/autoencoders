from enum import Enum
#from image_generators.simple_generator import SimpleGenerator
#from image_generators.body_centered_generator import BodyCenteredGenerator
from image_generators.striation_generator import StriationGenerator
from image_generators.image_generator import ImageGenerator

class GENERATORS(Enum):
   #SIMPLE = SimpleGenerator
   #BODY_CENTERED = BodyCenteredGenerator
   STRIATION = StriationGenerator
