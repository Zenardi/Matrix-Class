import math
from math import sqrt
import numbers
import numpy as np


def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        if len(self) == 1:
            return self[0][0]
        else:
            return self[0][0] * self[1][1] - self[0][1] * self[1][0]

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")
        sum = 0
        for i in range(len(self)):
            for j in range(len(self[0])):
                if(i == j):
                    sum += self[i][j]

        return sum


    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        return np.linalg.inv(self)


    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        return np.transpose(self)

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 

        matrixSum = []
        row = []
        for i in range(len(self)):
            for j in range(len(self[0])):
                row.append(self[i][j] + other[i][j])
            matrixSum.append(row)
            row = []
        return matrixSum

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        matrixNeg = []
        row = []
        for i in range(len(self)):
            for j in range(len(self[0])):
                row.append(-1 * self[i][j])
            matrixNeg.append(row)
            row = []
        return matrixNeg


    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        matrixSub = []
        row = []
        for i in range(len(self)):
            for j in range(len(self[0])):
                row.append(self[i][j] - other[i][j])
            matrixSub.append(row)
            row = []
        return matrixSub

    def get_column(self, column_number):
        column = []
        for i in range(len(self)):
            for j in range(len(self[i])):
                if(j==column_number):
                    column.append(self[i][j])
        return column

    def dot_product(self, other):
        s = 0
        for i in range(len(self)):
            s += self[i] * other[i]
        return s

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        m_rows = len(self)
        p_columns = len(other[0])
        # empty list that will hold the product of AxB
        result = []
        for i in range(m_rows):
            for j in range(p_columns):
                currentA = self[i]
                currentB =  get_column(other, j)
                dot_result = dot_product(currentA, currentB)
                row_result = []
                row_result.append(dot_result)
            result.append(row_result)

        return result



    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            pass
            #   
            # TODO - your code here
            #
            matrixRmul = []
            row = []
            for i in range(len(self)):
                for j in range(len(self[0])):
                    row.append(other * self[i][j])
                matrixRmul.append(row)
                row = []
            return matrixRmul
            
