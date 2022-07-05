# Script to plot C*-convex combinations in M_n(C)
#
# Author: Indrajit Ghosh
#
# Date: Jul 02, 2022
# Modified on: Jul 05, 2022
#

from manim import *
from linear_algebra import *
from manim import Matrix as ManimMatrix
from c_star_convexity import get_c_star_convex_combination


#####################################################
################ Initial Data #######################
#####################################################

# Change the matrix below
INPUT_MATRIX = Matrix([
    [-1-1j, 0, 2],
    [0, 3*1j, 0],
    [0, 0, 2 + 1j]
])    

LENGTH_OF_C_STAR_COMBINATION = 1
SHOW_CONVEX_HULL = False

####################################################
####################################################


class PlotEigenValues(Scene):
    
    def construct(self):

        # Creating the number plane
        plane = self.get_number_plane().add_coordinates()

        mat = INPUT_MATRIX # Input matrix

        # Plotting the eigenvalues of the matrix
        eigs_mat = self.get_eigenvalues_as_complex_point(plane, mat)

        eigval_color = RED  # Change the color here

        dots_mat = self.create_dots_from_complex_nums(eigs_mat, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=30).next_to(dots_mat[i], UP).set_color(eigval_color) for i in range(len(eigs_mat))
        ])

        if SHOW_CONVEX_HULL:
            polygon = Polygram(eigs_mat)
            self.add(plane, polygon, dots_mat, txt)
        else:
            self.add(plane, dots_mat, txt)


    def get_number_plane(self):

        number_plane = ComplexPlane(
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        return number_plane


    @staticmethod
    def get_eigenvalues_as_complex_point(complex_plane:ComplexPlane, mat:Matrix):
        """
        Returns the eigenvalues for plotting on the complex plane

        Returns
        -------
            `[np.array([lam.real, lam.imag, 0]) for lam in mat.eigen_values()]`
        """
        return [complex_plane.n2p(eig) for eig in mat.eigen_values()]

    @staticmethod
    def create_dots_from_complex_nums(complex_nums, **kwargs):
        return VGroup(*[Dot(z, **kwargs) for z in complex_nums])


class CStarConvexEigenValues(Scene):

    def construct(self):

        # Creating the number plane
        plane = PlotEigenValues.get_number_plane(self)

        D = INPUT_MATRIX # Input matrix
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = PlotEigenValues.get_eigenvalues_as_complex_point(plane, D)
        eigval_color = RED
        dots_D = PlotEigenValues.create_dots_from_complex_nums(eigs_D, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=35).next_to(dots_D[i], UP).set_color(eigval_color) for i in range(len(eigs_D))
        ])
        polygon = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = PlotEigenValues.get_eigenvalues_as_complex_point(plane, D_cc)
        dots_D_cc = PlotEigenValues.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW, radius=0.06)

        self.add(plane, polygon, dots_D, txt, dots_D_cc)


class CubeRootsCStarConvex(Scene):
    def construct(self):
        # Creating the number plane
        plane = PlotEigenValues.get_number_plane(self)

        # Unit circle
        unit_circle = Circle(radius=1, color=GREY)

        cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
        D = DiagonalMatrix(cube_roots_of_unity)
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = PlotEigenValues.get_eigenvalues_as_complex_point(plane, D)
        dots_D = PlotEigenValues.create_dots_from_complex_nums(eigs_D, color=RED)
        txt = VGroup(*[
            MathTex("1").next_to(eigs_D[0], RIGHT),
            MathTex("\\omega").next_to(eigs_D[1], UP),
            MathTex("\\omega^2").next_to(eigs_D[2], DOWN)
        ])
        triangle = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = PlotEigenValues.get_eigenvalues_as_complex_point(plane, D_cc)
        dots_D_cc = PlotEigenValues.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW, radius=0.06)

        self.add(plane, unit_circle, triangle, dots_D, txt, dots_D_cc)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()