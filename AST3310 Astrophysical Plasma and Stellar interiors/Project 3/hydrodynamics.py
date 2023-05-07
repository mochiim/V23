import FVis3 # visualiser
import scipy.constants as sc


class 2Dconvection:

    def __init__(self):

        """
        define variables
        """
        # constants
        self.k = sc.k           # Boltzmann constant [J K^-1]
        self.G = sc.G           # Newtonian constant of gravitation [m^3 kg^-1 s^-2]

        # model
        self.yaxis = 4e6        # length of box (vertical) [m]
        self.xaxis = 12e6       # width of box (horizontal) [m]
        self.Ny = 100           # number of boxes in vertical direction
        self.Nx = 300           # number of boxes in horizontal direction
        self.mu = .61           # mean mulecular weight
        
        # solar attributes
        self.T_sun = 5778                            # temperature of photosphere [K]
        self.P_sun = 1.8e4                           # pressure of photosphere [Pa]
        self.R_sun = 6.96e8                          # solar radius [m]
        self.M_sun = 1.989e30                        # solar mass [kg]
        self.g = - self.G*self.M_sun/self.R_sun**2   # constant gravitational acceleration in negative y-direction


    def initialise(self):

        """
        initialise temperature, pressure, density and internal energy
        """

    def timestep(self):

        """
        calculate timestep
        """

    def boundary_conditions(self):

        """
        boundary conditions for energy, density and velocity
        """

    def central_x(self, var):

        """
        central difference scheme in x-direction
        """

    def central_y(self, var):

        """
        central difference scheme in y-direction
        """

    def upwind_x(self, var, v):

        """
        upwind difference scheme in x-direction
        """

    def upwind_y(self, var, v):

        """
        upwind difference scheme in y-direction
        """

    def hydro_solver(self):

        """
        hydrodynamic equations solver
        """

        return dt

if __name__ == '__main__':
    # Run your code here