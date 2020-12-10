from PIL import Image, ImageOps

import numpy as np

if __name__ == '__main__':
    mat = np.loadtxt('output.txt', delimiter=' ')
    Image.fromarray(mat).show()