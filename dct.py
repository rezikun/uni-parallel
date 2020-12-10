import sys

print("success")
print(sys.version)

import numpy as np
from scipy.fftpack import idct, dct
from PIL import Image, ImageOps
from Pyro4 import expose
import copy

import scipy as sc

quantization_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                               [18, 21, 26, 66, 99, 99, 99, 99],
                               [24, 26, 56, 99, 99, 99, 99, 99],
                               [47, 66, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99],
                               [99, 99, 99, 99, 99, 99, 99, 99]]) * 3

THRESHOLD = 0.0005


#
# og_image = Image.open("image.png")
# gray_image = ImageOps.grayscale(og_image)
# image_matrix = np.asarray(gray_image.convert('L'))

def filterer(x):
    return x if abs(x) > THRESHOLD else 0


def dct2 (block):
  return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2 (block):
  return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def transform_cell(cell):
    cell = np.array(cell) - 128
    cell = dct2(cell)
    cell = np.divide(cell, quantization_table)
    cell = np.vectorize(filterer)(cell)
    return cell.tolist()


def detransform_cell(cell):
    cell = np.multiply(cell, quantization_table)
    cell = idct2(cell)
    cell = np.array(cell) + 128
    cell = np.vectorize(int)(cell)
    return cell.tolist()


def split_matrix_into_cells(matrix):
    cells_per_row = int(len(matrix[0]) / 8)
    cells_per_column = int(len(matrix) / 8)
    cells = [[0 for i in range(cells_per_row)] for i in range(cells_per_column)]
    for i in range(cells_per_column):
        for j in range(cells_per_row):
            cells[i][j] = matrix[i * 8:i * 8 + 8, j * 8:j * 8 + 8].copy().tolist()
    return cells

def trim_matrix(matrix):
    return matrix[0:int(len(matrix) / 8) * 8, 0:int(len(matrix[0]) / 8) * 8]

def encode_cell(cell):
    transformed_cell = transform_cell(cell)
    vector = []
    for i in range(0, len(transformed_cell)):
        for j in range(0, len(transformed_cell[0])):
            vector.append(transformed_cell[i][j])
    # codec = HuffmanCodec.from_data(vector)
    # return codec.encode(vector), codec
    return vector

# def decode_cell(code, codec, n, m):
def decode_cell(code, n=8, m=8):
    # decoded = codec.decode(code)
    matrix = [[0 for i in range(m)] for i in range(n)]
    for i in range(0, len(code)):
        matrix[int(i / m)][i % m] = code[i]
    return detransform_cell(matrix)

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.rows = 0
        self.columns = 0
        print("Inited")

    def tranform_to_np(self, matrix):
        transformed_matrix = None
        for i in range(len(matrix)):
            row = matrix[i].value
            row_concat = row[0]
            for j in range(1, len(row)):
                row_concat = np.concatenate((row_concat, np.array(row[j])), axis=1)
            if (transformed_matrix is None):
                transformed_matrix = copy.deepcopy(row_concat)
            else:
                transformed_matrix = np.concatenate((transformed_matrix, row_concat))
        return transformed_matrix

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))
        matrix = self.read_input()

        encoded_matrix = self.encode_matrix(matrix)

        decoded_matrix = self.decode_matrix(encoded_matrix)

        self.write_output(decoded_matrix)

        print("Job Finished")

    def encode_matrix(self, matrix):
        matrix = trim_matrix(matrix)
        cells = split_matrix_into_cells(matrix)
        encoded_matrix = [[0 for i in range(len(cells[0]))] for i in range(len(cells))]
        types = []
        for i in range(len(cells)):
            encoded_matrix[i] = self.workers[i % len(self.workers)].encode_row(cells[i])
        return encoded_matrix

    def decode_matrix(self, encoded_matrix):
        matrix = [[0 for i in range(self.columns//8)] for i in range(self.rows//8)]
        for i in range(len(encoded_matrix)):
            matrix[i] = self.workers[i % len(self.workers)].decode_row(encoded_matrix[i].value)
        return self.tranform_to_np(matrix)

    @staticmethod
    @expose
    def encode_row(row):
        encoded_row = []
        for cell in row:
            encoded_row.append(encode_cell(cell))
        return encoded_row

    @staticmethod
    @expose
    def decode_row(row):
        decoded_row = []
        for cell in row:
            decoded_row.append(decode_cell(cell))
        return decoded_row


    def read_input(self):
        input = np.loadtxt(self.input_file_name, dtype='f', delimiter=' ')
        self.rows = len(input)
        self.columns = len(input[0])
        return input

    def write_output(self, output):
        mat = np.matrix(output)
        np.savetxt(self.output_file_name, mat, fmt='%d')


if __name__ == '__main__':
    assert np.allclose (np.array([10]), idct2(dct2(np.array([10]))))
    worker = Solver()
    solver = Solver(workers=[worker], input_file_name='test.txt', output_file_name='result.txt')
    solver.solve()
