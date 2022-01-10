from linear_algebra import *

# Testing


def matrix_details(mat:Matrix):

    print(mat.prettify())

    info = f"""
    Number of rows : {mat.rows}
    Number of cols : {mat.cols}

    Trace : {mat.trace()}
    Determinant : {mat.det()}
    Rank : {mat.rank()}
    Eigenvalues : {mat.eigen_values()}

    Frobineus norm : {mat.lp_norm()}
    Maximum norm : {mat.max_norm()}
    Largest singular value : {mat.largest_singular_value()}
    Smallest singular value : {mat.smallest_singular_value()}
    Nuclear norm : {mat.nuclear_norm()}
    Maximum row sum : {mat.max_row_sum()}
    Minimum row sum : {mat.min_row_sum()}
    Maximum col sum : {mat.max_col_sum()}
    Minimum col sum : {mat.min_col_sum()}

    Symmetric : {mat.is_symmetric()}
    Self-adjoint : {mat.is_selfadjoint()}
    Normal : {mat.is_normal()}
    Unitary : {mat.is_unitary()}
    Projection : {mat.is_projection()}
    Isometry : {mat.is_isometry()}
    Positive : {mat.is_positive()}
    Positive definite : {mat.is_positive_definite()}
    Permutation : {mat.is_permutation()}


    """

    print(info)



def main():

    A = Matrix([[1, 0],
                [0, 1]])


    matrix_details(A)


if __name__ == '__main__':
    main()
    