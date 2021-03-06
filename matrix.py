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

        # 1 x 1 Matrix
        if self.h == 1:
            return self.g[0][0]

        # 2 x 2 Matrix:
        if self.h == 2:
            return self.g[0][0]*self.g[1][1] - self.g[0][1]*self.g[1][0]

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")
        sum = 0
        for i in range(self.h):
            for j in range(self.w):
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

        if self.h == 1:
            if self[0][0] == 0:
                raise(ValueError, "Value is 0, this matrix does not have an inverse.")
            else:
                return Matrix([[1/self[0][0]]])
        # calculate the inverse of a 2x2 matrix
        elif self.h == 2:
            I = identity(self.h)
            trace = self.trace()
            deter = self.determinant()
            # return an error if determinant is 0
            if deter == 0:
                raise ValueError('Determinant is 0, this matrix does not have an inverse.')
            # calculate the inverse of a 2x2 matrix
            else:
                return 1.0 / deter * ((trace * I) - self)


    def T(self):
        """
        Returns a transposed copy of this Matrix.
        # """
        transpose = zeroes(self.w,self.h)

        for i in range(transpose.h):
            for j in range(transpose.w):
                transpose[i][j] = self[j][i]
        return transpose

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

        matrixSum = zeroes(self.h, self.w)
        for i in range(self.h):
            for j in range(self.w):
                matrixSum[i][j] = (self[i][j] + other[i][j])
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
        # creates a self.h x self.w matrix of zeroes
        grid = zeroes(self.h, self.w)

        matrixNeg = zeroes(self.h, self.w)
        for i in range(self.h):
            for j in range(self.w):
                matrixNeg[i][j] = -1 * self.g[i][j]
        return matrixNeg


    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        matrixSum = zeroes(self.h, self.w)
        for i in range(self.h):
            for j in range(self.w):
                matrixSum[i][j] = (self[i][j] - other[i][j])
        return matrixSum

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        result = zeroes(self.h, other.w)
        for i in range(self.h):
            for j in range(other.w):
                for k in range(other.h):
                    result[i][j] += self.g[i][k] * other.g[k][j]
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
            for i in range(self.h):
                for j in range (self.w):
                    self[i][j] = other * self[i][j]
            return self

