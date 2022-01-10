# A class representing (complex) matrices
#
# Author: Indrajit Ghosh
#
# Date: Dec 24, 2021
#

from math import sqrt
import random
import sympy as sp
from scipy import linalg as LA
import numpy as np


class Matrix:

    """Class representing a matrix"""

    def __init__(self, default, order=None):
        """
            'default': 2D list
        """

        if isinstance(order, tuple):
            self.rows = order[0]
            self.cols = order[1]
        if isinstance(order, int):
            self.rows = self.cols = order
        else:
            self.rows = order[0] if order != None else len(default)
            self.cols = order[1] if order != None else len(default[0])

        self.order = (self.rows, self.cols)

        self.matrix = [[default] * self.cols for i in range(self.rows)] if order != None else default


    def __repr__(self):

        indent = '     '
        mat = '\n'

        for r in self.matrix:
            for j in range(self.cols):
                mat += indent + str(r[j]).center(7) + indent
            mat += '\n\n'

        return mat

    @staticmethod
    def I(n:int):
        """Identity Matrix of order n"""
        id = Matrix(order=(n, n), default=0)
        for i in range(n):
            id.matrix[i][i] = 1

        return id


    def __add__(self, other):

        M = Matrix(order=(self.rows, self.cols), default=0)

        # Addition with an object of type Matrix
        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    M.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]

        # Addition with scalars: we'll treat 'scalar' with corr 'scalar matrix'
        elif isinstance(other, (int, float, complex)):
            scalar_mat = Matrix(order=(self.rows, self.cols), default=0)
            for i in range(self.rows):
                scalar_mat.matrix[i][i] = other

            M = self.__add__(scalar_mat)

        return M


    # Addition by Matrix from right
    def __radd__(self, other):
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other):

        M = Matrix(order=(self.rows, self.cols), default=0)

        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    M.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]

            return M
        
        elif isinstance(other, (int, float, complex)):
            M.matrix = self.matrix
            for i in range(self.rows):
                M.matrix[i][i] = self.matrix[i][i] - other
            
            return M

    
    def __rsub__(self, other):
        return self.__sub__(other).schur_pdt(-1)


    # Matrix multiplications
    def __mul__(self, other):
        
        # Matrix Multiplication
        if isinstance(other, Matrix):

            if self.cols == other.rows:

                M = Matrix(order=(self.rows, other.cols), default=0)
                for i in range(self.rows):
                    for j in range(other.cols):
                        val = 0

                        for k in range(self.cols):
                            val += self.matrix[i][k] * other.matrix[k][j]

                        M.matrix[i][j] = val

                return M
            
            else:
                raise Exception("You cannot multiply them due to 'order' incompatibility!")


        # Scalar Multiplication
        if isinstance(other, (int, float, complex)):
            M = Matrix(order=(self.rows, self.cols), default=0)
            for i in range(self.rows):
                for j in range(self.cols):
                    M.matrix[i][j] = other * self.matrix[i][j]

            return M

    # Multiplication by Matrix from right
    def __rmul__(self, other):
        return self.__mul__(other)

    
    def __pow__(self, n:int):
        
        I = Matrix.I(self.rows)
        
        for _ in range(n):
            I *= self

        return I

    def transpose(self):
        """Transpose"""
        M = Matrix(default=0, order=(self.cols, self.rows))
        for i in range(M.rows):
            for j in range(M.cols):
                M.matrix[i][j] = self.matrix[j][i]
        
        return M

    def star(self):
        """Conjugate Transpose"""
        M = Matrix(default=0, order=(self.cols, self.rows))
        for i in range(M.rows):
            for j in range(M.cols):
                M.matrix[i][j] = complex(self.matrix[j][i]).conjugate()
        
        return M
    

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i = key[0] - 1
            j = key[1] - 1

            return self.matrix[i][j]

    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i = key[0] - 1
            j = key[1] - 1

            self.matrix[i][j] = value


    def schur_pdt(self, other):
        
        M = Matrix(order=(self.rows, self.cols), default=0)

        # Product with an object of type Matrix
        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    M.matrix[i][j] = self.matrix[i][j] * other.matrix[i][j]

        # Product with scalars
        elif isinstance(other, (int, float, complex)):
            for i in range(self.rows):
                for j in range(self.cols):
                    M.matrix[i][j] = other * self.matrix[i][j]

        return M

    def prettify(self):
        return sp.Matrix(self.matrix)

    def det(self):
        return np.linalg.det(np.array(self.matrix))

    def trace(self):
        ar = self.matrix
        return sum([ar[i][i] for i in range(self.rows)])

    def rank(self):
        return np.linalg.matrix_rank(np.array(self.matrix))

    def inverse(self):
        return Matrix(np.linalg.inv(np.array(self.matrix)))

    def charpoly(self):
        return self.prettify().charpoly(x)

    def eigen_values(self):
        return LA.eig(self.matrix)[0]

    
    @staticmethod
    def flatten(lst):
        """Take a multi dimensional list and flattens it"""
        return [item for sublist in lst for item in sublist]

    def vector_array(self):
        """Convert the matrix into a vector in C^{n^2}"""
        return self.flatten(self.matrix)


    def lp_norm(self, p=2):
        """
        Calculates l^p norm
        ||A||_l^p := [sum |A_ij|^p]^(1/p)
        """
        return sum(abs(el) ** p for el in self.vector_array()) ** (1 / p)  


    def max_norm(self):
        """
        Calculates max norm: max |A_ij|
        """  
        return max(abs(aij) for aij in self.vector_array())

    
    @staticmethod
    def check_permutaion_matrix(arr):
        """Checks whether arr represents a permutation matrix or not"""
        if all(sum(row) == 1 for row in arr):
            return all(sum(col) == 1 for col in zip(*arr))
        return False

    def is_permutation(self):
        return self.check_permutaion_matrix(self.matrix)


    def is_symmetric(self):
        """Checks whether it is equal to its transpose"""
        return self.matrix == self.transpose().matrix

    
    def is_selfadjoint(self):
        """Checks whether the matrix is hermitian or not"""
        return self.matrix == self.star().matrix

    def is_normal(self):
        """Checks for normality"""
        return self.__mul__(self.star()).matrix == self.star().__mul__(self).matrix 


    @staticmethod
    def convert_to_matrix_obj(arr):
        """Convert an 2 x 2 array into Matrix obj"""
        r, c = len(arr), len(arr[0])

        M = Matrix(order=(r, c), default=0)

        for i in range(r):
            for j in range(c):
                M[i + 1, j + 1] = arr[i][j]


        return M


class RandomMatrix(Matrix):
    def __init__(self, order, lower=-1, upper=1):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        arr = [[round(random.uniform(lower, upper), 2) for _ in range(cols)] for _ in range(rows)]

        super().__init__(default=arr)


class ScalarMatrix(Matrix):
    def __init__(self, order:int=3, scalar=1):

        arr = [[0] * order for _ in range(order)]
        for i in range(order):
            for j in range(order):
                arr[i][i] = scalar

        super().__init__(default=arr)


class BasisMatrix(Matrix):
    def __init__(self, i:int=2, j:int=2, order=3):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        arr = [[0] * cols for _ in range(rows)]
        for ii in range(rows):
            for jj in range(cols):
                if (ii + 1, jj + 1) == (i, j):
                    arr[ii][jj] = 1
        super().__init__(default=arr)


def get_matrix_basis(dim:int=2):
    """Generate standard basis in M_n(C)"""
    basis = {}
    for ii in range(dim):
        for jj in range(dim):
            basis.setdefault('E' + str(ii + 1) + str(jj + 1), BasisMatrix(i= ii + 1, j = jj + 1, order= dim))
    return basis


class JordanBlock(Matrix):
    """Jordan Block matrix"""

    def __init__(self, size:int, symbol):

        mat = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if j == i + 1:
                    mat[i][j] = 1
                if j == i:
                    mat[i][i] = symbol
        
        super().__init__(default=mat)


class HilbertMatrix(Matrix):
    """Hilbert Matrix"""

    def __init__(self, size:int=5):

        mat = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                mat[i][j] = 1 / (i +  j + 1)

        super().__init__(default=mat)


class VandermondeMatrix(Matrix):
    """Vandermonde Matrix"""

    # TODO: Complete it
    def __init__(self, symbol=2, size:int=5):
    
        mat = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                mat[i][j] = symbol ** (j)

        super().__init__(default=mat)


class Vector(Matrix):
    """Class Representing a Matrix"""
    def __init__(self, default, dim:int=None):
        """
            'default': 1D list
        """
        if isinstance(default, list):
            vals = [[v] for v in default] 
            super().__init__(default=vals, order=None)
        else:
            super().__init__(default, order=(dim, 1))

        self.norm = sqrt(self.dot(self)[1, 1].real)

    
    def __getitem__(self, key):
        return super().__getitem__((key, 1))
    
    def __setitem__(self, key, value):
        return super().__setitem__((key, 1), value)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return super().__mul__(other)
        elif isinstance(other, (int, float, complex)):
            return Vector(super().__mul__(other).vector_array())


    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise ZeroDivisionError("You cannot divide by zero!")
            else:
                return Vector(self.__mul__(1 / other).vector_array())

        else:
            raise ValueError("You cannot divide by a vector!")

    
    def unit(self):
        """Returns the unit vector along that direction"""
        arr = [val / self.norm for val in self.vector_array()]
        return Vector(arr)

    
    def star(self):
        return super().star()

    
    def as_projection_operator(self):
        """Projection operator onto the 1D space spanned by that vector"""
        n = 1 / (self.norm ** 2)
        return super().__mul__(self.star()).__mul__(n)

    def dot(self, other):
        """Hermitian inner product"""
        if isinstance(other, Vector):
            return other.star() * self


class BasisVector(Vector):

    def __init__(self, dim: int = 3, i:int=1):
        arr = [0 for _ in range(dim)]
        for id, e in enumerate(arr):
            if id + 1 == i:
                arr[id] = 1 
        super().__init__(default=arr, dim=dim)


class RandomVector(Vector):
    def __init__(self, dim:int = 3, lower=-1, upper=1, desired_norm:float=None):

        arr = [round(random.uniform(lower, upper), 2) for _ in range(dim)]
        nr = sqrt(sum(abs(el) ** 2 for el in arr))

        if desired_norm == None:
            super().__init__(default=arr)
        else:
            arr = [(desired_norm * el) / nr for el in arr]
            super().__init__(default=arr)


# SymPy Symbols
x, y, z, k = sp.symbols('x,y,z, k')
alpha = sp.symbols('alpha')
beta = sp.symbols('beta')
gamma = sp.symbols('gamma')
delta = sp.symbols('delta')
omega = sp.symbols('omega')
sigma = sp.symbols('sigma')
theta = sp.symbols('theta')
pi = sp.symbols('pi')
lam = sp.symbols('lambda')
kappa = sp.symbols('kappa')
rho = sp.symbols('rho')
chi = sp.symbols('chi')
epsilon = sp.symbols('epsilon')
Delta = sp.symbols('Delta')
Omega = sp.symbols('omega')


class SymbolicMatrix:

    def __init__(self, default, order: tuple = None):

        if isinstance(default, Matrix):
            self.rows = default.rows
            self.cols = default.cols

            self.mat_for_indexing = default.matrix

        else:
            if isinstance(order, tuple):
                self.rows = order[0]
                self.cols = order[1]

            elif isinstance(order, int):
                self.rows = self.cols = order
            
            else:
                self.rows = len(default)
                self.cols = len(default[0])

            self.mat_for_indexing = [[default] * self.cols for i in range(self.rows)] if order != None else default
        
        self.order = (self.rows, self.cols)
        self.matrix = sp.Matrix(self.mat_for_indexing)


    def __repr__(self):
        return str(self.matrix)


    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            scalar = SymbolicMatrix(ScalarMatrix(order=self.rows, scalar=other))
            return SymbolicMatrix(np.array(self.matrix.__add__(scalar.matrix)))

        return SymbolicMatrix(np.array(self.matrix.__add__(other.matrix)))
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            scalar = SymbolicMatrix(ScalarMatrix(order=self.rows, scalar=other))
            return SymbolicMatrix(np.array(self.matrix.__sub__(scalar.matrix)))
        return SymbolicMatrix(np.array(self.matrix.__sub__(other.matrix)))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            scalar = SymbolicMatrix(ScalarMatrix(order=self.rows, scalar=other))
            return SymbolicMatrix(np.array(self.matrix.__mul__(scalar.matrix)))
        return SymbolicMatrix(np.array(self.matrix.__mul__(other.matrix)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, n):
        return SymbolicMatrix(np.array(self.matrix.__pow__(n)))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i = key[0] - 1
            j = key[1] - 1

            return self.mat_for_indexing[i][j]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i = key[0] - 1
            j = key[1] - 1

            self.mat_for_indexing[i][j] = value
            self.matrix = sp.Matrix(self.mat_for_indexing)
    

    def det(self):
        return self.matrix.det()

    def rank(self):
        return self.matrix.rank()

    def charpoly(self):
        return self.matrix.charpoly()

    def eigen_values(self):
        return self.matrix.eigenvals()

    def inverse(self):
        return SymbolicMatrix(np.array(self.matrix.inv()))

    def echelon_form(self):
        return self.matrix.echelon_form()
    
    