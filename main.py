from linear_algebra import *
from datetime import datetime
import subprocess, os


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


def create_tex_file(mat:Matrix, **kwargs):
    """
    Generates a `tex` file with the content `mat.get_tex()`
    """
    tex_str = r"""

\documentclass{article}
\usepackage{amsmath}

\begin{document}
\thispagestyle{empty}

    """

    tex_str += mat.get_tex(**kwargs)

    tex_str += "\n\n" + r"""\end{document}""" + "\n\n"

    timestamp = str(datetime.now().timestamp()).replace('.', '')
    texfilename = timestamp + '.tex'
    pdffilename = timestamp + '.pdf'
    texFile = "./media/Tex/" + texfilename


    with open(texFile, "w") as f:
        f.write(tex_str)
    
    os.chdir("./media/Tex/")
    subprocess.run(["pdflatex", texfilename])
    os.system('clear')

    print(f"""
        The tex file lives here:\n\t {texFile}
        """)
    subprocess.run(["xdg-open", pdffilename])


def main():

    A = HaarDistributedUnitary(size=5)

    # matrix_details(A)
    create_tex_file(A)



if __name__ == '__main__':
    main()
    