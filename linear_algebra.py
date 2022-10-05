# Important classes for Linear Algebra!
#
# Author: Indrajit Ghosh
#
# Date: Dec 24, 2021
# Modified on: Jul 06, 2022
#

# TODO: Add `Convex Optimization Solver`: Source: "https://www.cvxpy.org/"
# TODO: Add  code for `Moore–Penrose inverse` in Matrix

import sympy as sp
import numpy as np


###########################################################################################
###########################################################################################
################                 Numerical Matrices             ###########################
###########################################################################################
###########################################################################################


class Matrix:

    """
    Class representing a complex matrix
    Author: Indrajit Ghosh

    Parameters
    ----------
        `default`: `2Dlist` or 2D `np.ndarray` or `int/float/complex`
        `order`: `int` or `tuple`= None (optional, needed only if `default` is `int/float/complex`)

    Returns
    -------
        ``Matrix```


    Examples
    -------
        >>> A = Matrix(
                        [
                            [1.2, 0.3],
                            [0.1, 0]
                        ]
                    )

        >>> A
        >>>
        >>> arr = np.arange(0, 9, 1).reshape(3, 3)
        >>> mat = Matrix(default=7, order=(3, 2))
        >>> mat.prettify()

    """

    def __init__(self, default, order=None, _complex:bool=True):
        """
            'default': 2D list (or numpy 2D array)
            'order': tuple or int
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

        if _complex:
            self.matrix = np.full((self.rows, self.cols), default, dtype='complex') if order != None else np.asarray(default, dtype='complex')
        else:
            self.matrix = np.full((self.rows, self.cols), default) if order != None else np.asarray(default)


    def __repr__(self):

        class_name = "Matrix"

        # arr = [[self.matrix[i][j] for j in range(self.cols)] for i in range(self.cols)]
        arr_str = np.array2string(self.matrix, separator=', ')

        rep = class_name + '(\n' + arr_str + '\n)'

        return rep
        

    def __str__(self):

        indent = '     '
        mat = '\n'

        for r in self.matrix:
            for j in range(self.cols):
                mat += indent + str(r[j]).center(7) + indent
            mat += '\n\n'

        return mat


    def get_tex(self, **kwargs):
        """
        Converts the ```Matrix``` object into a LaTeX expression

        Parameter
        ---------
            `round_off`: int
        
        Returns
        -------
            LaTeX expression for the Matrix

        Example
        -------

        >>> from linear_algebra import *
        >>> A = Matrix([
                        [5.73+3*1j, 34.2+1j],
                        [0.8, 3.3]
                        ])
        >>> A.get_tex(round_off=2)

        >>>    \begin{pmatrix}
                   5.73+3.0i   & 34.2+i  \\
                   0.8   & 3.3  \\
               \end{pmatrix}

        """

        matrix = self.matrix

        matrix_str = [['' for _ in matrix[0]] for _ in matrix]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix_str[i][j] += Matrix.format_complex_to_str(matrix[i][j], **kwargs)

        tex_command = r"""\[""" + "\n" + r"""\begin{pmatrix}""" + "\n"
        for row in matrix_str:
            tex_command +=  '    ' + self.add_ampersand_and_linebreak_tex(row)

        tex_command += '\\end{pmatrix}\n' + r"""\]"""

        return tex_command


    @staticmethod
    def add_ampersand_and_linebreak_tex(arr):
        """Accepts a 1D list and add '&' and '\\' chars properly"""
        _tex = ''.join(x + '   & ' for x in arr)[:-3] + '\\\\' + '\n'
        return _tex

    
    @staticmethod
    def format_complex_to_str(z:complex=1 - 1j, round_off:int=2):
        """
        A function that formats a ```complex()``` into a ```str``` of the form `a+bi`

        Parameters
        -----------
            `z`: complex
            `round_off`: int

        Returns
        -------
            `str` of the form `z.real+z.imag i`

        """

        if z.imag == 0:
            # Real numbers
            return str(int(z.real)) if z.real.is_integer() else str(np.round(z.real, round_off))

        else:
            if z.real == 0:
                # Purely imaginary
                if z.imag == 1:
                    return "i"
                elif z.imag == -1:
                    return "-i"
                else:
                    return str(int(z.imag)) + "i" if z.imag.is_integer() else str(np.round(z.imag, round_off)) + "i"

            else:
                # Complex numbers
                if z.imag > 0:
                    if z.imag == 1:
                        return str(np.round(z.real, round_off)) + "+" + "i"
                    else:
                        return str(np.round(z.real, round_off)) + "+" + str(np.round(z.imag, round_off)) + "i"
                else:
                    if z.imag == -1:
                        str(np.round(z.real, round_off)) + "-" + "i"
                    else:
                        return str(np.round(z.real, round_off)) + "-" + str(np.round(np.abs(z.imag), round_off)) + "i"

    
    def __eq__(self, other):
        """Equality between matrices"""
        if isinstance(other, Matrix):
            return np.allclose(self.matrix, other.matrix, equal_nan=True)

    
    def __ne__(self, other: object):
        if isinstance(other, Matrix):
            return not self.__eq__(other)


    def __add__(self, other):

        # Addition with an object of type Matrix
        if isinstance(other, Matrix) and type(other) != Vector:
            return Matrix(self.matrix + other.matrix)
        else:
            # Addition with scalars: we'll treat 'scalar' with corr 'scalar matrix'
            if isinstance(other, (int, float, complex)):
                return Matrix(self.matrix + (other * np.eye(self.rows, self.cols)))
            else:
                return NotImplemented


    # Addition by Matrix from right
    def __radd__(self, other):
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other):

        if isinstance(other, Matrix) and type(other) != Vector:
            return Matrix(self.matrix - other.matrix)
        else:
            if isinstance(other, (int, float, complex)):
                return Matrix(self.matrix - (other * np.eye(self.rows, self.cols)))
            else:
                return NotImplemented

    
    def __rsub__(self, other):
        return self.__sub__(other).schur_pdt(-1)


    # Matrix multiplications
    def __mul__(self, other):
        
        # Matrix Multiplication
        if isinstance(other, Matrix) and type(other) != Vector:
            return Matrix(np.matmul(self.matrix, other.matrix))

        else:
            # Scalar Multiplication
            if isinstance(other, (int, float, complex)):
                return Matrix(other * self.matrix)
            else:
                return NotImplemented


    # Multiplication by Matrix from right
    def __rmul__(self, other):
        return self.__mul__(other)

    # Division by a scalar
    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise ZeroDivisionError("You cannot divide by zero!")
            else:
                return Matrix(self.matrix / other)

        else:
            raise ValueError("You cannot divide by a matrix!")

    
    def __pow__(self, n:int):
        return Matrix(np.linalg.matrix_power(self.matrix, n))

    
    def get_upper_left_block(self, size:int=2):
        """
        Returns the upper-left `size` by `size` block submatrix 
        """
        block_arr = self.matrix[0:size, 0:size]

        return Matrix(block_arr)
        

    def transpose(self):
        """Transpose"""
        return Matrix(self.matrix.T)

    def star(self):
        """Conjugate Transpose"""
        arr = self.matrix
        arr_conj_trans = arr.conjugate().T
        
        return Matrix(arr_conj_trans)
    

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

        # Product with an object of type Matrix
        if isinstance(other, Matrix):
            return Matrix(self.matrix * other.matrix)

        # Product with scalars
        elif isinstance(other, (int, float, complex)):
            return Matrix(other * self.matrix)


    def prettify(self, decimal_places:int=2):
        return sp.Matrix(np.round(self.matrix, decimals=decimal_places))

    def det(self):
        return np.linalg.det(self.matrix)

    def trace(self, offset:int=0):
        """
        The sum along its diagonal with the given offset
        is returned, i.e., the sum of elements ``self.matrix[i,i+offset]`` for all i.
        """
        return np.trace(self.matrix, offset=offset)

    def rank(self):
        return np.linalg.matrix_rank(self.matrix)

    def inverse(self):
        return Matrix(np.linalg.inv(self.matrix))

    def pseudoinverse(self):
        """
        Calculates the Moore-Penrose inverse
        """
        return Matrix(np.linalg.pinv(self.matrix))

    def charpoly(self):
        t = sp.symbols('t')
        return self.prettify().charpoly(t)

    def eigen_values(self):
        return np.linalg.eigvals(self.matrix)

    def qr_decomposition(self):
        """Return Q and R of the QR Decomposition"""
        Q, R = np.linalg.qr(self.matrix)

        return Matrix(Q), Matrix(R)

    
    @staticmethod
    def flatten(lst):
        """Take a multi dimensional list and flattens it"""
        return [item for sublist in lst for item in sublist]

    def vector_array(self):
        """Convert the matrix into a vector in C^{n^2}"""
        return self.matrix.flatten()


    def col_sum(self, col:int=1):
        """
        Calculate the sum of the given col
        """
        return np.sum(self.matrix, axis=0)[(col-1) % self.cols]

    def row_sum(self, row:int=1):
        """
        Calculate the sum of the given row
        """
        return np.sum(self.matrix, axis=1)[(row-1) % self.rows]

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

    def max_row_sum(self):
        """
            return max(sum(abs(x) along rows))
        """
        return np.linalg.norm(self.matrix, ord=np.inf)

    def min_row_sum(self):
        return np.linalg.norm(self.matrix, ord=-np.inf)

    def max_col_sum(self):
        return np.linalg.norm(self.matrix, ord=1)

    def min_col_sum(self):
        return np.linalg.norm(self.matrix, ord=-1)

    def largest_singular_value(self):
        """
         the singular values, or s-numbers of a compact operator T: X → Y acting 
         between Hilbert spaces X and Y, are the square roots of non-negative 
         eigenvalues of the self-adjoint operator T*T (where T* denotes the adjoint of T)
        """
        return np.linalg.norm(x=self.matrix, ord=2)

    def smallest_singular_value(self):
        """
         the singular values, or s-numbers of a compact operator T: X → Y acting 
         between Hilbert spaces X and Y, are the square roots of non-negative 
         eigenvalues of the self-adjoint operator T*T (where T* denotes the adjoint of T)
        """
        return np.linalg.norm(x=self.matrix, ord=-2)

    def nuclear_norm(self):
        """
        This is known as the 'nuclear norm'
        Sum of all singular values
        """
        return np.linalg.norm(x=self.matrix, ord='nuc')

    def is_square(self):
        """
        Checks whether a matrix is square or not
        """
        return self.rows == self.cols
    
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
        return np.array_equal(self.matrix, self.transpose().matrix, equal_nan=True)

    
    def is_selfadjoint(self):
        """Checks whether the matrix is hermitian or not"""
        return np.allclose(self.matrix, self.star().matrix)

    def is_normal(self):
        """Checks for normality"""
        return np.allclose(self.__mul__(self.star()).matrix, self.star().__mul__(self).matrix)

    def is_isometry(self):
        """Checks for isometries"""
        I = np.eye(self.rows)
        if np.allclose(self.star().__mul__(self).matrix, I):
            return True
        else:
            return False

    def is_unitary(self):
        """Checks for unitary matrices"""
        if np.allclose(np.abs(self.eigen_values()), np.ones(self.rows)):
            return True
        else:
            return False


    def is_positive(self):
        """
        In Matrix analysis it is sometimes called 'positive semi definite':
        A Matrix is positive iff it is normal with non-negative eigenvalues
        """
        if self.is_normal():
            if np.all(self.eigen_values() >= 0):
                return True
            elif np.allclose(self.eigen_values()[self.eigen_values() < 0], 0):
                return True
            else:
                return False
        else:
            return False
        

    def is_positive_definite(self):
        """
        All eigenvalues are > 0 and normal
        """
        return np.all(self.eigen_values() > 0) and self.is_selfadjoint()


    def is_projection(self):
        """Self-adjoint idempotent"""
        return self.is_selfadjoint() and self.__eq__(self.__pow__(2))

    
    def is_partial_isometry(self):
        """
        V is partial isometry iff VV* is projection
        """
        return self.__mul__(self.star()).is_projection()

    
    def is_row_stochastic(self):
        """
        Checks whether it is `row` stochastic, i.e., non-negative square matrix 
        with each row summing to 1
        """
        arr = self.matrix
        ones = np.ones(shape=(len(arr),))
        nonNeg = np.all(arr >= 0)
        square = arr.shape[0] == arr.shape[1]
        row_sum_one = np.allclose(np.sum(arr, axis=0), ones)

        return square and nonNeg and row_sum_one

    def is_col_stochastic(self):
        """
        Checks whether it is `col` stochastic, i.e., non-negative square matrix 
        with each col summing to 1
        """
        arr = self.matrix
        ones = np.ones(shape=(len(arr),))
        nonNeg = np.all(arr >= 0)
        square = arr.shape[0] == arr.shape[1]
        col_sum_one = np.allclose(np.sum(arr, axis=1), ones)

        return square and nonNeg and col_sum_one

    def is_stochastic(self):
        """
        Checks whether it is doubly stochastic, i.e., non-negative square matrix 
        with each row and col summing to 1
        """
        return self.is_row_stochastic() and self.is_col_stochastic()




###########################################################################################
###########################################################################################
################                 Complex Vectors                ###########################
###########################################################################################
###########################################################################################

class Vector(Matrix):
    """
    Class Representing a complex Vector

    Parameters
    ----------
        `default`: `list` or `np.ndarray` or `int/float/complex`
        `dim`: `int`= None (optional, needed only if `default` is `int/float/complex`)

    Returns
    -------
        ```Vector```

    Examples
    --------
        >>> Vector(default=3.2, dim=2)
            
            (3.2+0j)     

            (3.2+0j) 

        >>> v = Vector([1, 0, 2j])
        >>> v

            (1+0j)     

              0j       
      
              2j  

        >>> v[1]

            (1+0j)

        >>> v.norm()

            2.23606797749979

        >>> RandomVector(dim=2, desired_norm=2)

    """

    def __init__(self, default, dim:int=None, **kwargs):
        """
            'default': 1D array
        """
        if isinstance(default, (list, np.ndarray)):
            vals = np.array(default).reshape(len(default), 1) 
            super().__init__(default=vals)
        else:
            super().__init__(default, order=(dim, 1), **kwargs)


    def __repr__(self):
        
        class_name = "Vector"

        arr_str = np.array2string(self.vector_array(), separator=', ')

        rep = class_name + '(' + arr_str + ')'

        return rep

    
    def __getitem__(self, key):
        return super().__getitem__((key, 1))
    
    def __setitem__(self, key, value):
        return super().__setitem__((key, 1), value)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.matrix + other.matrix)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Vector):
            return self.__add__(other.__mul__(-1))

        if isinstance(other, (int, float, complex)):
            return Vector(self.matrix - other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            return Vector(2 - self.matrix)
        if isinstance(other, Vector):
            return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Vector(super().__mul__(other).vector_array())

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            return Vector(np.matmul(other.matrix, self.matrix))
        
        elif isinstance(other, (int, float, complex)):
            return self.__mul__(other)


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
        arr = [val / self.norm() for val in self.vector_array()]
        return Vector(arr)

    def component_along(self, v):
        """Return the component of 'self' along the vector v"""
        return (self.dot(v) / v.dot(v)) * v
    
    def star(self):
        return super().star()

    def dot(self, other):
        """Hermitian inner product"""
        if isinstance(other, Vector):
            z_matrix = other.star() * self
            return z_matrix[1]
        else:
            return NotImplementedError("`dot` product can only be performed between two Vectors!")

    def norm(self):
        """Calculates the Hermitian norm of the Vector"""
        return np.sqrt(self.dot(self).real)


    def is_orthogonal(self, other):
        if np.allclose(self.dot(other), 0):
            return True
        else:
            return False


    def tensor(self, w):
        """
        Return the tensor product of 'self' and 'w'
        """
        return super().__mul__(w.star())

    
    def as_projection_operator(self):
        """Projection operator onto the 1D space spanned by that vector"""
        n = 1 / (self.norm() ** 2)
        return super().__mul__(self.star()).__mul__(n)



#################################################################################
#################################################################################
################## Classes based on `Matrix` and `Vector` ######################
#################################################################################
#################################################################################


class BlockDiagonalMatrix(Matrix):
    """
    Class representing a block-diagonal matrix

    Parameters
    ----------
        `blocks`: list
            ```list``` of blocks; each of these block could be `np.ndarray` or `Matrix`

    Returns
    -------
        `Matrix` class object
    """
    I2 = [
        [1, 0],
        [0, 1]
    ]
    I3 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    def __init__(self, blocks:list=[I2, I3], **kwargs):
        arr = self.block_matrix(blocks)
        super().__init__(default=arr, **kwargs)


    @staticmethod
    def block_matrix(blocks:list):
        """
        Creates an block-diagonal array from a list of `blocks`

        Returns
        -------
            `2D-list`
        """
        rows = []
        cols = []

        matrices = []

        for block in blocks:
            if isinstance(block, (int, float, complex)):
                rows.append(1)
                cols.append(1)
                matrices.append([[block]])
            
            elif isinstance(block, (np.ndarray, list)):
                rows.append(len(block))
                cols.append(len(block[0]))
                matrices.append(block)

            elif isinstance(block, Matrix):
                rows.append(len(block.matrix))
                cols.append(len(block.matrix[0]))
                matrices.append(block.matrix)

        mat = [[0 for _ in range(sum(cols))] for _ in range(sum(rows))]


        dr, dc = 0, 0
        for block in matrices:
            for i in range(sum(rows)):
                for j in range(sum(cols)):
                    if (dr <= i < dr + len(block)) and (dc <= j < dc + len(block[0])):
                        mat[i][j] += block[i - dr][j - dc]

            dr += len(block)
            dc += len(block[0])
        
        return mat

    @staticmethod
    def amplify(mat:Matrix, num_of_times:int=2):
        """
        Amplify a `Matrix` by given number and returns it

        Parameters
        ----------
            `mat`: Matrix
            `num_of_times`: int

        Returns
        -------
            `BlockDiagonalMatrix([mat,...,mat])`
    
        """
        ls = [mat for _ in range(num_of_times)]

        return BlockDiagonalMatrix(ls)


class Identity(Matrix):
    """Identity Matrix of order n"""
    def __init__(self, size:int=3, **kwargs):
        super().__init__(default=np.eye(size), _complex=False, **kwargs)


class RandomMatrix(Matrix):
    """
    Generates a random matrix whose entries are `Uniform[a, b)` distributed

    Parameters
    ----------
        `order`: int or tuple
        `lower`: float; 
            this is the value of `a`. default will be `-1`
        `upper`: float; 
            this is the value of `b`. default will be `1`
        `_complex`:bool
            default ```True```
            if `_complex` is `True` then the class will generate a random 
            matrix with complex entries otherwise real entries.
    """
    def __init__(self, order=2, lower=-1, upper=1, _complex:bool=True, **kwargs):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        a, b = lower, upper
        choice = np.random.choice(['Real', 'Complex']) if _complex else 'Real'
        if choice == 'Real':
            arr = self.generate_real_random_array(size=(rows, cols), lower=a, upper=b)
        else:
            arr = self.generate_complex_random_array(size=(rows, cols), lower=a, upper=b)

        super().__init__(default=arr, _complex=_complex, **kwargs)


    @staticmethod
    def generate_real_random_array(size:tuple=(2, 2), lower:float=-1, upper:float=1):
        a, b = lower, upper
        ran_arr = (b - a) * np.random.random_sample(size=(size[0], size[1])) + a # Uniform[a, b); b>a
        return ran_arr

    @staticmethod
    def generate_complex_random_array(size:tuple=(2, 2), lower:float=-1, upper:float=1):
        a, b = lower, upper
        ran_arr1 = (b - a) * np.random.random_sample(size=(size[0], size[1])) + a # Uniform[a, b); b>a
        ran_arr2 = (b - a) * np.random.random_sample(size=(size[0], size[1])) + a # Uniform[a, b); b>a
        ran_arr = ran_arr1 + 1j * ran_arr2
        return ran_arr


class RandomGaussianMatrix(Matrix):
    """
    Generates a random matrix whose entries are `Normal(m, sigma)` distributed

    Parameters
    ----------
        `order`: int or tuple
        `m`: float; 
            mean of the Normal distribution
        `sigma`: float; 
            standard deviation of the Normal distribution
        `_complex`:bool
            default ```True```
            if `_complex` is `True` then the class will generate a random 
            matrix with complex entries otherwise real entries.
    """
    def __init__(self, order=2, mean=0, sigma=1, _complex:bool=True, **kwargs):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        choice = np.random.choice(['Real', 'Complex']) if _complex else 'Real'
        if choice == 'Real':
            arr = self.random_real_normal_sample(shape=(rows, cols), m=mean, sigma=sigma)
        else:
            arr = self.random_complex_normal_sample(shape=(rows, cols), m=mean, sigma=sigma)

        super().__init__(default=arr, _complex=_complex, **kwargs)


    @staticmethod
    def random_real_normal_sample(shape:tuple=(3, 2), m:float=0, sigma:float=1):
        """
        Parameters
        ----------
            `tuple`:(int, ..., int)

        Returns
        --------
            `ndarray`: array of floating-point samples from the standard normal distribution
        """
        return m + sigma * np.random.randn(*shape)
    
    @staticmethod
    def random_complex_normal_sample(**kwargs):
        """
        Complex normal sampling
        """
        arr = RandomGaussianMatrix.random_real_normal_sample(**kwargs) + 1j * RandomGaussianMatrix.random_real_normal_sample(**kwargs)
        return arr


class RandomBooleanMatrix(Matrix):
    """
    Class for a random boolean matrix

    Parameters
    ----------
        `order`: tuple or int
        `p`: float (must be in [0, 1])

    Returns
    --------
        `Matrix`

    """
    def __init__(self, order=3, p:float=0.5, **kwargs):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        arr = np.random.rand(rows, cols) > p
        super().__init__(default=arr, _complex=False, **kwargs)


class DiagonalMatrix(Matrix):
    """
    Class for representing a diagonal matrix
    """
    def __init__(self, diagonal=[1, 2, 3], **kwargs):
        arr = np.diag(np.array(diagonal))
        super().__init__(default=arr, **kwargs)
    
        
class ScalarMatrix(Matrix):
    def __init__(self, order:int=3, scalar=1, **kwargs):

        arr = np.zeros((order, order))
        for i in range(order):
            for _ in range(order):
                arr[i][i] = scalar

        super().__init__(default=arr, **kwargs)


class HaarDistributedUnitary(Matrix):
    """
    A class that gives a matrix-valued U(n) random variable.

    Parameter
    ---------
    `size`: integer
            dimension of the matrix

    Returns
    -------
    An object of the class `Matrix`

    Reference
    ----------
    ... F. Mezzadri, "How to generate random matrices from the classical compact groups", 
            :arXiv:"https://arxiv.org/pdf/math-ph/0609050.pdf".

    Example
    -------
    >>> from linear_algebra import HaarDistributedUnitary
    >>> H = HaarDistributedUnitary(size=2)
    >>> H

    """
    def __init__(self, size:int=2, **kwargs):

        random_arr = self.unitary_group_generator(size=size)

        super().__init__(default=random_arr, **kwargs)


    @staticmethod
    def unitary_group_generator(size:int=2):
        """
        A matrix-valued U(n) random variable.

        The `size` parameter specifies the order `n`.

        Parameter
        ---------
        `size`: integer
                dimension of the matrix

        """

        N = size # initializing the order of our matrices

        # Step 1: generate `normally (0, 1) distributed` NxN real matrices
        A, B = np.random.normal(size=(N, N), loc=0, scale=1), np.random.normal(size=(N, N), loc=0, scale=1)

        # Step 2: creating a `normally distributed` complex NxN matrix
        Z = A + 1j * B

        # Step 3: calculating QR decomposition of Z
        Q, R = np.linalg.qr(Z)

        # Step 4: computing the diagonal matrix Lambda := diag(R_ii / abs(R_ii))
        diagonal_of_lam = np.diag(R) / np.abs(np.diag(R)) # extracting the diagonal
        lam = np.diag(diagonal_of_lam) # computing lamda

        # Step 5: computing Q'
        Q1 = np.dot(Q, lam)

        return Q1


class RandomOrthogonalMatrix(Matrix):
    """
    A class that gives a matrix-valued O(n) random variable.

    Parameter
    ---------
    `size`: integer
            dimension of the matrix

    Returns
    -------
    An object of the class `Matrix`

    Reference
    ----------
    ... F. Mezzadri, "How to generate random matrices from the classical compact groups", 
            :arXiv:"https://arxiv.org/pdf/math-ph/0609050.pdf".

    Example
    -------
    >>> from linear_algebra import HaarDistributedUnitary
    >>> O = RandomOrthogonalMatrix(size=2)
    >>> O

    """
    def __init__(self, size:int=2, **kwargs):

        random_arr = self.orthogonal_group_generator(size=size)

        super().__init__(default=random_arr, _complex=False, **kwargs)


    @staticmethod
    def orthogonal_group_generator(size:int=2):
        """
        A matrix-valued O(n) random variable.

        The `size` parameter specifies the order `n`.

        Parameter
        ---------
        `size`: integer
                dimension of the matrix

        """

        N = size # initializing the order of our matrices

        # Step 1: generate `normally (0, 1) distributed` NxN real matrix
        Z = np.random.normal(size=(N, N), loc=0, scale=1)

        # Step 3: calculating QR decomposition of Z
        Q, R = np.linalg.qr(Z)

        # Step 4: computing the diagonal matrix Lambda := diag(R_ii / abs(R_ii))
        diagonal_of_lam = np.diag(R) / np.abs(np.diag(R)) # extracting the diagonal
        lam = np.diag(diagonal_of_lam) # computing lamda

        # Step 5: computing Q'
        Q1 = np.dot(Q, lam)

        return Q1


class RandomDoublyStochasticMatrix(Matrix):
    """
    A class that gives a random DoublyStochasticMatrix, which is a 
    matrix of nonnegative real numbers, each of whose rows and columns sums to 1

    Parameter
    ---------
    `size`: integer
            dimension of the matrix

    Returns
    -------
    An object of the class `Matrix`


    Algorithm
    ---------
        1. Generate a RandomOrthogonalMatrix, say, O = [O_ij]
        2. Return the matrix [|O_ij|^2]

    Example
    -------
    >>> from linear_algebra import RandomDoublyStochasticMatrix
    >>> O = RandomDoublyStochasticMatrix(size=3)
    >>> O

    """
    def __init__(self, size:int=3, **kwargs):

        ortho_arr = RandomOrthogonalMatrix.orthogonal_group_generator(size)
        stocha_arr = np.vectorize(lambda z: np.abs(z) ** 2)(ortho_arr)

        super().__init__(default=stocha_arr, _complex=False, **kwargs)
        

class RandomDensityMatrix(Matrix):
    """
    Class representing a random density matrix

    Positive elements of the C* algebra M_n(C) with trace 1
    euivalently an positive definite matrix of trace 1

    Parameters
    ----------
        `size`:int  
            order of the matrix
        `lower`:float
            default `-1`
        `upper`:float
            default `1`
        `_complex`:bool
            default ```True```
            if `_complex` is `True` then the class will generate a density 
            matrix with complex entries otherwise real entries.

    Returns
    -------
        `Matrix` class object

    Example
    -------
    >>> from linear_algebra import RandomDensityMatrix
    >>> rho = RandomDensityMatrix(size=3)
    >>> rho.prettify()

    """
    def __init__(self, size:int=2, lower=-1, upper=1, _complex:bool=True, **kwargs):

        a = lower
        b = upper

        choice = np.random.choice(['Real', 'Complex']) if _complex else 'Real'
        if choice == 'Real':
            density_mat = self.generate_real_density_matrix(size=size, lower=a, upper=b)
        else:
            density_mat = self.generate_complex_density_matrix(size=size, lower=a, upper=b)

        super().__init__(default=density_mat, _complex=_complex, **kwargs)

    
    @staticmethod
    def generate_real_density_matrix(size:int=2, lower=-1, upper=1):
        """
        Positive trace 1 matrices in M_n(C)
        """
        a = lower
        b = upper
        ran_arr = (b - a) * np.random.random_sample(size=(size, size)) + a # Uniform[a, b); b>a
        ran_arr_trans = ran_arr.T # taking transpose
        rho = np.dot(ran_arr_trans, ran_arr) # creating positive matrix

        ran_density_mat = rho / np.trace(rho) # making trace 1

        return ran_density_mat

    @staticmethod
    def generate_complex_density_matrix(size:int=2, lower=-1, upper=1):
        """
        Positive trace 1 matrices in M_n(C)
        """
        a = lower
        b = upper
        ran_arr1 = (b - a) * np.random.random_sample(size=(size, size)) + a # Uniform[a, b); b>a
        ran_arr2 = (b - a) * np.random.random_sample(size=(size, size)) + a # Uniform[a, b); b>a
        ran_arr = ran_arr1 + 1j * ran_arr2 # generating random complex matrix

        ran_arr_conj_trans = ran_arr.conjugate().T # taking conjugate transpose

        rho = np.dot(ran_arr_conj_trans, ran_arr) # creating positive matrix
        ran_density_mat = rho / np.trace(rho) # making trace 1

        return ran_density_mat


class BasisMatrix(Matrix):
    """
    Gives the (rectangular) elementary basis matrices ```E_{ij}``` 
    """
    def __init__(self, i:int=2, j:int=2, order=3, **kwargs):

        if isinstance(order, tuple):
            rows = order[0]
            cols = order[1]
        if isinstance(order, int):
            rows = cols = order

        arr = np.zeros((rows, cols))
        for ii in range(rows):
            for jj in range(cols):
                if (ii + 1, jj + 1) == (i, j):
                    arr[ii][jj] = 1
        super().__init__(default=arr, _complex=False, **kwargs)

    @staticmethod
    def get_matrix_basis(dim:int=2):
        """Generate standard basis in M_n(C)"""
        basis = {}
        for ii in range(dim):
            for jj in range(dim):
                basis.setdefault('E' + str(ii + 1) + str(jj + 1), BasisMatrix(i= ii + 1, j = jj + 1, order= dim))
        return basis


class HilbertMatrix(Matrix):
    """Hilbert Matrix"""

    def __init__(self, size:int=5, **kwargs):

        mat = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                mat[i][j] = 1 / (i +  j + 1)

        super().__init__(default=mat, **kwargs)


class JordanBlock(Matrix):
    """Jordan Block matrix"""

    def __init__(self, size:int, scalar, **kwargs):

        mat = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if j == i + 1:
                    mat[i][j] = 1
                if j == i:
                    mat[i][i] = scalar
        
        super().__init__(default=mat, **kwargs)


class VandermondeMatrix(Matrix):
    """
    A class representing Vandermonde Matrix, a matrix with the terms 
    of a geometric progression in each row: an m x n matrix

    Parameters
    ----------
        `scalars`:list
            default= [1, 2, 3, 4, 5]
        `cols`:int
            default=4
    
    Returns
    -------
        `Matrix`

    """
    def __init__(self, scalars:list=[1, 2, 3, 4, 5], cols:int=4, **kwargs):
        mat = self.vandermonde(scalars=scalars, cols=cols)
        super().__init__(default=mat, **kwargs)

    
    @staticmethod
    def vandermonde(scalars:list=[1, 2, 3, 4, 5], cols:int=4):
        """
        Creates an vandermonde array

        Parameters
        ----------
            `scalars`:list
            `cols`:int

        Returns
        -------
            `np.ndarray`
        """
        arr = np.array([[x ** i for i in range(cols)] for x in scalars])
        return arr
    
    @staticmethod
    def roots_of_unity(n:int=3):
        """
        Returns
        -------
            `list`: the list of nth roots of unity: z_k = exp((2 pi k i) / n); k = 0, 1, ..., n-1

        """
        roots = [np.exp((2*np.pi*k*1j)/n) for k in range(n)]
        return roots


class PauliMatrix(Matrix):
    def __init__(self, j:int=1, **kwargs):
        """
            n = 1, 2, 3
        """
        super().__init__(default=self.pauli(j), **kwargs)

    @staticmethod
    def kronecker_delta(i, j):
        return 1 if i == j else 0
    
    def pauli(self, j):
        sigma_j = np.array([[self.kronecker_delta(j, 3), self.kronecker_delta(j, 1) - 1j * self.kronecker_delta(j, 2)],
                       [self.kronecker_delta(j, 1) + 1j * self.kronecker_delta(j, 2), - self.kronecker_delta(j, 3)]])
        return sigma_j


class BasisVector(Vector):

    def __init__(self, dim: int = 3, i:int=1, **kwargs):
        arr = np.zeros(dim)
        for id, e in enumerate(arr):
            if id + 1 == i:
                arr[id] = 1 
        super().__init__(default=arr, dim=dim, _complex=False, **kwargs)


class RandomVector(Vector):
    """
    Generates a random `Vector` whose entries are `Uniform[a, b)` distributed

    Parameters
    ----------
        `dim`: int
        `lower`: float; 
            this is the value of `a`. default will be `-1`
        `upper`: float; 
            this is the value of `b`. default will be `1`
        `_complex`:bool
            default ```True```
            if `_complex` is `True` then the class will generate a random 
            vector with complex entries otherwise real entries.
    """

    def __init__(self, dim:int = 3, lower=-1, upper=1, desired_norm:float=None, _complex:bool=True, **kwargs):

        a = lower
        b = upper
        if _complex:
            arr1 = (b - a) * np.random.random_sample(size=dim) + a # Uniform[a, b); b>a
            arr2 = (b - a) * np.random.random_sample(size=dim) + a # Uniform[a, b); b>a
            arr = arr1 + 1j * arr2
        else:
            arr = (b - a) * np.random.random_sample(size=dim) + a # Uniform[a, b); b>a
        arr = np.round(arr, decimals=2) # rounding off

        nr = np.linalg.norm(arr)

        if desired_norm == None:
            super().__init__(default=arr, _complex=_complex, **kwargs)
        else:
            arr = (desired_norm * arr) / nr
            super().__init__(default=arr, _complex=_complex, **kwargs)


class RandomQuantumState(RandomVector):
    """Random vector of norm one"""

    def __init__(self, dim: int = 3, **kwargs):
        super().__init__(dim, desired_norm=1, **kwargs)


###########################################################################################
###########################################################################################
################                System of Equations             ###########################
###########################################################################################
###########################################################################################

class SystemOfLinearEquations:
    """class representing a system of linear equation"""

    def __init__(self, coef_mat:Matrix, b:Vector):
        self.coef_mat = coef_mat
        self.b = b

    def __repr__(self):
        s = "\nSystem of Linear Equation:\n\n"
        s += f"Coefficient matrix:\n{self.coef_mat.matrix}\n"
        s += f"\nVector b:\n {self.b.matrix}\n"

        return s

    def solve(self):
        A = self.coef_mat.matrix
        b = self.b.matrix

        return Vector(np.linalg.solve(A, b).reshape(self.coef_mat.cols, 1))

    def satisfied_by(self, v:Vector):
        """Check whether a vector satisfies the system or not"""
        w = self.coef_mat * v
        if np.allclose(w.matrix, self.b.matrix):
            return True
        else:
            return False



###########################################################################################
###########################################################################################
################                  Symbolic Matrices             ###########################
###########################################################################################
###########################################################################################

# SymPy Symbols
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
    """Class to represent a Matrix with arbitrary symbols"""

    def __init__(self, default, order: tuple = None):

        if isinstance(default, Matrix):
            self.rows = default.rows
            self.cols = default.cols

            self.indexing_matrix = default.matrix

        else:
            if isinstance(order, tuple):
                self.rows = order[0]
                self.cols = order[1]

            elif isinstance(order, int):
                self.rows = self.cols = order
            
            else:
                self.rows = len(default)
                self.cols = len(default[0])

            self.indexing_matrix = [[default] * self.cols for i in range(self.rows)] if order != None else default
        
        self.order = (self.rows, self.cols)
        self.matrix = sp.Matrix(self.indexing_matrix)


    def __repr__(self):
        return str(self.matrix)


    def __eq__(self, other):
        """Equality between matrices"""
        if isinstance(other, SymbolicMatrix):
            return self.matrix == other.matrix

    
    def __ne__(self, other: object):
        if isinstance(other, SymbolicMatrix):
            return not self.__eq__(other)


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

            return self.indexing_matrix[i][j]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i = key[0] - 1
            j = key[1] - 1

            self.mat_for_indexing[i][j] = value
            self.matrix = sp.Matrix(self.mat_for_indexing)


    def prettify(self):
        return self.matrix

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


class SymbolicJordanBlock(SymbolicMatrix):
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


class SymbolicVandermondeMatrix(SymbolicMatrix):
    """
    A class representing Symbolic Vandermonde Matrix, a `symbolic` matrix with the terms 
    of a geometric progression in each row: an m x n matrix

    Parameters
    ----------
        `symbols`:list
            default= [1, omega, beta]
        `cols`:int
            default=4
    
    Returns
    -------
        `SymbolicMatrix`

    """

    def __init__(self, symbols=[1, omega, beta], cols:int=4):

        mat = self.symbolic_vandermonde(scalars=symbols, cols=cols)

        super().__init__(default=mat)

    
    @staticmethod
    def symbolic_vandermonde(scalars:list=[1, omega, alpha], cols:int=4):
        """
        Creates an vandermonde array

        Parameters
        ----------
            `scalars`:list
            `cols`:int

        Returns
        -------
            `np.ndarray`
        """
        arr = np.array([[x ** i for i in range(cols)] for x in scalars])
        return arr
