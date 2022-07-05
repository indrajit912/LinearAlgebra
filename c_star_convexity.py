# Codes related to C-star Convexity
#
# Author: Indrajit Ghosh
#
# Date: Jul 05, 2022
#

from linear_algebra import *
from convex_hull import *


def convert_1D_array_of_complex_nums_into_2D_array(arr:np.array=np.array([1+1j, 1j, -3])):
    """
    It converts the array of complex number: np.array([z1, z2, ..., zn]) and 
    returns:
                np.array([
                    [z1.real, z1.imag],
                    [z2.real, z2.imag],
                            :
                            :
                    [zn.real, zn.imag]
                ])

    
    Parameter
    ---------
        `np.array`: shape ```(N,)```, i.e. 1D array
    Returns
    -------
        `np.array`: shape ```(N, 2)```

    """
    return np.array([(z.real, z.imag) for z in arr])


def extreme_points_of_convex_hull(complex_numbers:np.array=np.array([1+1j, 1j, -3])):
    """
    Returns the extreme points of the given `complex_numbers`

    Example
    -------
    >>> extreme_points_of_convex_hull()

            array([-3.+0.j,  1.+1.j,  0.+1.j])
       
    """
    arr = convert_1D_array_of_complex_nums_into_2D_array(complex_numbers)
    vecs = get_vertices_of_convex_hull(arr)

    return np.array([complex(v[0], v[1]) for v in vecs])


def check_complex_num_is_in_convex_hull(complex_nums=np.array([0, 1, 1j]), given_num:complex=1+1j):
    """
    Checks whether a given complex number is inside the ConvexHull of `complex_nums

    Return
    ------
        `Bool`
    """

    vecs = convert_1D_array_of_complex_nums_into_2D_array(complex_nums)
    p = (given_num.real, given_num.imag)

    return is_inside_convex_hull(vecs, p)


def get_c_star_convex_combination(mat:Matrix, length:int=2):
    """
    Returns a random C-star convex combination of the given matrix of 
    certain `length`.

    Returns
    -------
        `Matrix`

    Algorithm
    ----------
        We are going to find a C*-convex coefficients {T_1,...,T_k} subset of M_n(C)
        (i.e., they satisfy \sum_{i=1}^{k}T_i^* T_i = I)

        For this we note that for a `nk` by `nk` unitary matrix, say U, (in the block 
        matrix form: consisting of `n` by `n` blocks in a kxk matrix)

                U = |T_11, T_12, ..., T_1k|
                    |T_21, T_22, ..., T_2k|;    T_ij \in M_n(C)
                    |       .....         |
                    |T_k1, T_k2, ..., T_kk|

        
        each row of U is a C*-convex coefficients {T_i1, ..., T_ik} for i = 1, 2, ..., k


        Step 1: Take a `n x n` matrix A 
        Step 2: Generate a random `nk x nk` unitary U
        Step 3: Compute B := U* diag(A, ..., A) U   ### k many A's in diag()
        Step 4: Upper left k x k block of B is the the C*-convex combination of A:
            T_11* A T_11 + T_12* A T_12 + ... + T_1k* A T_1k
    """

    if not mat.is_square():
        raise Exception("The matrix is not square!")
    else:
        n, k = mat.rows, length

        U = HaarDistributedUnitary(size=n*k) # Generating a random unitary of size `nk by nk`
        mat_amplified = BlockDiagonalMatrix.amplify(mat=mat, num_of_times=k) # amplifying the matrix

        M = U.star() * mat_amplified * U

        # Returning the upper-left block of size `n`
        return M.get_upper_left_block(size=n)


def main():

    pass

    


if __name__ == '__main__':
    main()