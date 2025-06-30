import sys
import os
# Add the parent directory of image_generators to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from image_generators.AtomP6 import AtomP6

def main():
    print("Testing AtomP6 class...")
    atom_p6 = AtomP6()
    atom_p6.generate()

if __name__ == "__main__":
    main()