# Codes related to C-star Convexity
#
# Author: Indrajit Ghosh
#
# Date: Jul 05, 2022
#

from linear_algebra import *


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