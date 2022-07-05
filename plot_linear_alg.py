# Script to plot stuffs from linear_algebra
#
# Author: Indrajit Ghosh
#
# Date: Jul 02, 2022
#

from manim import *
from linear_algebra import *
from c_star_convexity import get_c_star_convex_combination


class SpectralDistribution(Scene):

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

        # Generating matrix
        cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
        D = DiagonalMatrix(cube_roots_of_unity)

        D_cc = get_c_star_convex_combination(D, 2)

        # Plotting the eigenvalues of the matrix D
        eigs_D = self.get_eigenvalues_as_complex_num(D)
        dots_D = self.create_dots_from_complex_nums(eigs_D, color=RED)
        triangle = Polygram(eigs_D)

        # Plotting for the c-star convex combination
        eigs_D_cc = self.get_eigenvalues_as_complex_num(D_cc)
        dots_D_cc = self.create_dots_from_complex_nums(eigs_D_cc, color=YELLOW)

        self.add(number_plane, unit_circle, dots_D, triangle, dots_D_cc)
    

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


def main():
    print('hello world!')


if __name__ == '__main__':
    main()