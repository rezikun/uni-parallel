import numpy as np
from PIL import Image, ImageOps
import copy

def open_as_np(filename):
    im = Image.open(filename)
    gray_image = ImageOps.grayscale(im)
    return np.asarray(gray_image.convert('L'))

def multiply(image_matrix, n, m):
    result = copy.deepcopy(image_matrix)
    for i in range(n):
        result = np.concatenate(result, result)
    for i in range(m):
        result = np.concatenate(result, result, axis=1)
    return result

def save_image(image_matrix, filename):
    mat = np.matrix(image_matrix)
    np.savetxt(filename, mat, fmt='%d')

if __name__ == '__main__':
    image_matrix = open_as_np("big.jpg")
    # image_matrix = multiply(image_matrix, 2, 2)
    save_image(image_matrix, "big.txt")