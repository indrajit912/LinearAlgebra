# Script to plot stuffs from linear_algebra
#
# Author: Indrajit Ghosh
#
# Date: Jul 02, 2022
#

from manim import *
from linear_algebra import *


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

        # Generating random unitary matrix
        R = RandomMatrix(order=20)
        H = HaarDistributedUnitary(size=10)
        P = RandomDensityMatrix(size=10)

        mat = P # Change here

        # Plotting the eigenvalues of the matrix
        complex_nums = [np.array([eig.real, eig.imag, 0]) for eig in mat.eigen_values()]
        dots = VGroup(*[Dot(num, color=RED, radius=0.05) for num in complex_nums])

        self.add(number_plane, unit_circle, dots)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()