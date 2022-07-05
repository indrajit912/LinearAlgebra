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
from c_star_convexity import get_c_star_convex_combination, extreme_points_of_convex_hull
from convex_hull import generate_uniform_sample_points


#####################################################
################ Initial Data #######################
#####################################################

# Change the matrix below
INPUT_MATRIX = Matrix([
    [-1-1j, 0, 2],
    [0, 3*1j, 0],
    [0, 0, 2 + 1j]
]) 

# INPUT_MATRIX = RandomMatrix(order=10)

LENGTH_OF_C_STAR_COMBINATION = 2
SHOW_CONVEX_HULL = True

####################################################
####################################################


class ConvexHull(Scene):
    def construct(self):

        # Creating the number plane
        plane = self.get_number_plane().add_coordinates()

        com_nums = self.generate_random_complex_numbers(lower=-3, upper=3)
        com_points = self.convert_complex_nums_to_points(plane, com_nums)

        dots_to_plot = self.create_dots_from_complex_nums(com_points, color=ORANGE)
        
        if SHOW_CONVEX_HULL:
            extreme_points_to_plot = self.get_extreme_points_to_plot(plane, com_nums)
            hull_path = Polygram(extreme_points_to_plot, fill_color=YELLOW, fill_opacity=0.1, stroke_color=PURPLE_B)
            self.add(plane, hull_path, dots_to_plot)
        else:
            self.add(plane, dots_to_plot)
    
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
    def generate_random_complex_numbers(size:int=10, **kwargs):
        """
        Returns
        -------
        """
        points = generate_uniform_sample_points(shape=(size, 2), **kwargs)

        return np.array([complex(p[0], p[1]) for p in points])


    @staticmethod
    def convert_complex_nums_to_points(complex_plane, complex_nums):
        return [complex_plane.n2p(z) for z in complex_nums]


    @staticmethod
    def create_dots_from_complex_nums(complex_nums, **kwargs):
        return VGroup(*[Dot(z, **kwargs) for z in complex_nums])

    @staticmethod
    def get_extreme_points_to_plot(complex_plane:ComplexPlane, complex_nums):
        """
        Returns
        -------
            `list`: [np.array([z1.real, z1.imag, 0]), .... , np.array([zn.real, zn.imag, 0])]
        """
        extreme_com_nums = extreme_points_of_convex_hull(complex_numbers=complex_nums)

        return [complex_plane.n2p(z) for z in extreme_com_nums]
        


class PlotEigenValues(Scene):
    
    def construct(self):

        # Creating the number plane
        plane = ConvexHull.get_number_plane(self).add_coordinates()

        mat = INPUT_MATRIX # Input matrix

        # Plotting the eigenvalues of the matrix
        eigen_vals = mat.eigen_values()
        eigs_points = ConvexHull.convert_complex_nums_to_points(plane, eigen_vals)

        eigval_color = RED  # Change the color here
        eigs_dots_to_plot = ConvexHull.create_dots_from_complex_nums(eigs_points, color=eigval_color)

        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=30).next_to(eigs_dots_to_plot[i], UP).set_color(eigval_color) for i in range(len(eigs_points))
        ])

        if SHOW_CONVEX_HULL:
            extreme_points_to_plot = ConvexHull.get_extreme_points_to_plot(plane, eigen_vals)
            hull_path = Polygram(extreme_points_to_plot, fill_color=YELLOW, fill_opacity=0.1, stroke_color=PURPLE_B)
            self.add(plane, hull_path, eigs_dots_to_plot, txt)
        else:
            self.add(plane, eigs_dots_to_plot, txt)


    @staticmethod
    def get_eigenvalues_as_complex_point(complex_plane:ComplexPlane, mat:Matrix):
        """
        Returns the eigenvalues for plotting on the complex plane

        Returns
        -------
        """
        return ConvexHull.convert_complex_nums_to_points(complex_plane, mat.eigen_values())


class CStarConvexEigenValues(Scene):

    def construct(self):

        # Creating the number plane
        plane = ConvexHull.get_number_plane(self)

        D = INPUT_MATRIX # Input matrix
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigen_vals_D = D.eigen_values()
        eigs_points_D = ConvexHull.convert_complex_nums_to_points(plane, eigen_vals_D)

        eigval_color = RED  # Change the color here
        eigs_dots_to_plot_D = ConvexHull.create_dots_from_complex_nums(eigs_points_D, color=eigval_color)
        txt = VGroup(*[
            MathTex("\\lambda_{0}".format(i+1), font_size=35).next_to(eigs_points_D[i], UP).set_color(eigval_color) for i in range(len(eigs_points_D))
        ])


        # Plotting for the c-star convex combination
        eigen_vals_D_cc = D_cc.eigen_values()
        eigs_points_D_cc = ConvexHull.convert_complex_nums_to_points(plane, eigen_vals_D_cc)
        eigs_dots_to_plot_D_cc = ConvexHull.create_dots_from_complex_nums(eigs_points_D_cc, color=YELLOW, radius=0.06)


        extreme_points_to_plot = ConvexHull.get_extreme_points_to_plot(plane, eigen_vals_D)
        hull_path = Polygram(extreme_points_to_plot, fill_color=YELLOW, fill_opacity=0.1, stroke_color=PURPLE_B)

        self.add(plane, hull_path, eigs_dots_to_plot_D, txt, eigs_dots_to_plot_D_cc)


class CubeRootsCStarConvex(Scene):
    def construct(self):
        # Creating the number plane
        plane = ConvexHull.get_number_plane(self)

        cube_roots_of_unity = VandermondeMatrix.roots_of_unity()
        D = DiagonalMatrix(cube_roots_of_unity)
        D_cc = get_c_star_convex_combination(D, LENGTH_OF_C_STAR_COMBINATION) # c_star combination

        # Plotting the eigenvalues of the matrix D
        eigen_vals_D = D.eigen_values()
        eigs_points_D = ConvexHull.convert_complex_nums_to_points(plane, eigen_vals_D)

        eigval_color = RED  # Change the color here
        eigs_dots_to_plot_D = ConvexHull.create_dots_from_complex_nums(eigs_points_D, color=eigval_color)

        txt = VGroup(*[
            MathTex("1").next_to(eigs_points_D[0], RIGHT),
            MathTex("\\omega").next_to(eigs_points_D[1], UP),
            MathTex("\\omega^2").next_to(eigs_points_D[2], DOWN)
        ])
        triangle = Polygram(eigs_points_D, fill_color=YELLOW, fill_opacity=0.1, stroke_color=PURPLE_B)

        # Plotting for the c-star convex combination
        eigen_vals_D_cc = D_cc.eigen_values()
        eigs_points_D_cc = ConvexHull.convert_complex_nums_to_points(plane, eigen_vals_D_cc)
        eigs_dots_to_plot_D_cc = ConvexHull.create_dots_from_complex_nums(eigs_points_D_cc, color=YELLOW, radius=0.06)

        # Unit circle
        unit_circle = Circle(color=PINK, stroke_width=2)

        self.add(plane, unit_circle, triangle, eigs_dots_to_plot_D, txt, eigs_dots_to_plot_D_cc)


def main():
    print('hello world!')


if __name__ == '__main__':
    main()