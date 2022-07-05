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
    """

    if not mat.is_square():
        raise Exception("The matrix is not square!")
    else:
        n, k = mat.rows, length

        U = HaarDistributedUnitary(size=n*k) # Generating a random unitary of size `nk by nk`
        mat_amplified = BlockDiagonalMatrix.amplify(mat=mat, num_of_times=k) # amplifying the matrix

        M = U * mat_amplified * U.star()

        # Returning the upper-left block of size `n`
        return M.get_upper_left_block(size=n)


def main():

    pass

    


if __name__ == '__main__':
    main()