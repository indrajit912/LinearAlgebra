# Script to plot stuffs from linear_algebra
#
# Author: Indrajit Ghosh
#
# Date: Jul 02, 2022
#

from cmath import e
import csv
from manim import *
from linear_algebra import *
from c_star_convexity import get_c_star_convex_combination


#####################################################
################ Initial Data #######################
#####################################################

cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
INPUT_MATRIX = DiagonalMatrix(cube_roots_of_unity)      # Change the matrix here

LENGTH_OF_C_STAR_COMBINATION = 50

####################################################
####################################################


class CStarConvexEigenValues(Scene):

    def construct(self):

        # Creating the number plane
        number_plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        # Unit circle
        unit_circle = Circle(radius=1, color=GREY)

        D = INPUT_MATRIX # Input matrix
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = self.get_eigenvalues_as_complex_num(D)
        eigval_color = RED
        dots_D = self.create_dots_from_complex_nums(eigs_D, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=35).next_to(dots_D[i], UP).set_color(eigval_color) for i in range(len(eigs_D))
        ])
        polygon = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = self.get_eigenvalues_as_complex_num(D_cc)
        dots_D_cc = self.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW)

        self.add(number_plane, polygon, dots_D, txt, dots_D_cc)
    

    @staticmethod
    def get_eigenvalues_as_complex_num(mat:Matrix):
        """
        Returns the eigenvalues for plotting on the complex plane

        Returns
        -------
            `[np.array([lam.real, lam.imag, 0]) for lam in mat.eigen_values()]`
        """
        return [np.array([eig.real, eig.imag, 0.0]) for eig in mat.eigen_values()]

    @staticmethod
    def create_dots_from_complex_nums(complex_nums, **kwargs):
        return VGroup(*[Dot(num, radius=0.05, **kwargs) for num in complex_nums])


class CubeRootsCStarConvex(Scene):
    def construct(self):
        # Creating the number plane
        number_plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        # Unit circle
        unit_circle = Circle(radius=1, color=GREY)

        cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
        D = DiagonalMatrix(cube_roots_of_unity)
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = CStarConvexEigenValues.get_eigenvalues_as_complex_num(D)
        dots_D = CStarConvexEigenValues.create_dots_from_complex_nums(eigs_D, color=RED)
        txt = VGroup(*[
            MathTex("1").next_to(eigs_D[0], RIGHT),
            MathTex("\\omega").next_to(eigs_D[1], UP),
            MathTex("\\omega^2").next_to(eigs_D[2], DOWN)
        ])
        triangle = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = CStarConvexEigenValues.get_eigenvalues_as_complex_num(D_cc)
        dots_D_cc = CStarConvexEigenValues.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW)

        self.add(number_plane, unit_circle, triangle, dots_D, txt, dots_D_cc)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()