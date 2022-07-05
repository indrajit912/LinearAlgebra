# Script to plot C*-convex combinations in M_n(C)
#
# Author: Indrajit Ghosh
#
# Date: Jul 02, 2022
# Modified on: Jul 05, 2022
#

from manim import *
from linear_algebra import *
from c_star_convexity import get_c_star_convex_combination


#####################################################
################ Initial Data #######################
#####################################################

# Change the matrix below
INPUT_MATRIX = Matrix([
    [-1-1j, 0, 2],
    [0, 5*1j, 0],
    [0, 0, 2 + 1j]
])      


LENGTH_OF_C_STAR_COMBINATION = 3

####################################################
####################################################


class PlotEigenValues(Scene):
    
    def construct(self):

        # Creating the number plane
        plane = self.get_number_plane()
        

        mat = INPUT_MATRIX # Input matrix

        # Plotting the eigenvalues of the matrix
        eigs_mat = self.get_eigenvalues_as_complex_num(mat)
        eigval_color = RED
        dots_mat = self.create_dots_from_complex_nums(eigs_mat, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=35).next_to(dots_mat[i], UP).set_color(eigval_color) for i in range(len(eigs_mat))
        ])
        polygon = Polygram(eigs_mat)

        self.add(plane, polygon, dots_mat, txt)


    def get_number_plane(self):

        number_plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.4
            }
        )

        return number_plane


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



class CStarConvexEigenValues(Scene):

    def construct(self):

        # Creating the number plane
        plane = PlotEigenValues.get_number_plane(self)

        D = INPUT_MATRIX # Input matrix
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = PlotEigenValues.get_eigenvalues_as_complex_num(D)
        eigval_color = RED
        dots_D = PlotEigenValues.create_dots_from_complex_nums(eigs_D, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=35).next_to(dots_D[i], UP).set_color(eigval_color) for i in range(len(eigs_D))
        ])
        polygon = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = PlotEigenValues.get_eigenvalues_as_complex_num(D_cc)
        dots_D_cc = PlotEigenValues.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW)

        self.add(plane, polygon, dots_D, txt, dots_D_cc)



class CubeRootsCStarConvex(Scene):
    def construct(self):
        # Creating the number plane
        number_plane = PlotEigenValues.get_number_plane(self)

        # Unit circle
        unit_circle = Circle(radius=1, color=GREY)

        cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
        D = DiagonalMatrix(cube_roots_of_unity)
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigs_D = PlotEigenValues.get_eigenvalues_as_complex_num(D)
        dots_D = PlotEigenValues.create_dots_from_complex_nums(eigs_D, color=RED)
        txt = VGroup(*[
            MathTex("1").next_to(eigs_D[0], RIGHT),
            MathTex("\\omega").next_to(eigs_D[1], UP),
            MathTex("\\omega^2").next_to(eigs_D[2], DOWN)
        ])
        triangle = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = PlotEigenValues.get_eigenvalues_as_complex_num(D_cc)
        dots_D_cc = PlotEigenValues.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW)

        self.add(number_plane, unit_circle, triangle, dots_D, txt, dots_D_cc)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()