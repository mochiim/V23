import FVis3 # visualiser
import scipy.constants as sc


class hydrodynamics:
    def __init__(self):

        """
        define variables
        """
        # constants
        self.k = sc.k           # Boltzmann constant [J K^-1]
        self.G = sc.G           # Newtonian constant of gravitation [m^3 kg^-1 s^-2]
        self.m_u = sc.m_u       # atomic mass unit [kg]

        # model
        self.yaxis = 4e6                # length of box (vertical) [m]
        self.xaxis = 12e6               # width of box (horizontal) [m]
        self.Ny = 100                   # number of boxes in vertical direction
        self.Nx = 300                   # number of boxes in horizontal direction
        self.dx = self.xaxis/self.Nx    #  
        self.dy = self.yaxis/self.Ny    #
        self.mu = .61                   # mean mulecular weight
        
        # solar attributes
        self.T_sun = 5778                            # temperature of photosphere [K]
        self.P_sun = 1.8e4                           # pressure of photosphere [Pa]
        self.rho_sun = 2.3e-4                        # density of photosphere [kg m^-3]
        self.R_sun = 6.96e8                          # solar radius [m]
        self.M_sun = 1.989e30                        # solar mass [kg]
        self.g = - self.G*self.M_sun/self.R_sun**2   # constant gravitational acceleration in negative y-direction

        # arrays to store computed values
        self.T = np.zeros([self.Nx, self.Ny])       # temperature
        self.P = np.zeros([self.Nx, self.Ny])       # pressure
        self.rho = np.zeros([self.Nx, self.Ny])     # density
        self.u = np.zeros([self.Nx, self.Ny])       # horizontal velocity
        self.w = np.zeros([self.Nx, self.Ny])       # vertical velocity
        self.e = np.zeros([self.Nx, self.Ny])       # internal energy

    def initialise(self, Gauss = False):

        """
        initialise temperature, pressure, density and internal energy
        """
        nabla = 2.5 + .01

        self.T[:] = self.T_sun - self.mu * self.m_u * self.g * nabla * self.dy / self.k
        self.P[:] = self.P_sun * (self.T / self.T_sun) ** (1 / nabla)
        self.e[:] = 3/2 * self.P
        self.rho[:] = P * self.mu * self.m_u / ( self.k * self.T)
        self.u[:] = 0

    def timestep(self):

        """
        calculate timestep
        """

    def boundary_conditions(self):

        """
        boundary conditions for energy, density and velocity
        """
        # vertical boundary: vertical velocity
        self.w[:0] = 0
        self.w[:-1] = 0

        # vertical boundary: horizontal velocity
        self.u[:0] = (- self.u[:2] + 4*self.u[:1]) / 3
        self.u[:-1] = (- self.u[:-2] + 4 * self.u[:-1]) / 3

        # vertical boundary: internal energy
        self.e[:, 0]    = ( - self.e[:, 2] + 4 * self.e[:, 1] ) / ( 3 - 2 * self.mu * self.m_u * self.g / ( self.k * self.T[:, 0] ) * self.Dy )
        self.e[:, -1]   = ( - self.e[:, -3] + 4 * self.e[:, -2] ) / ( 3 + 2 * self.mu * self.m_u * self.g / ( self.k * self.T[:, -1] ) * self.Dy )
        
        # vertical boundary: density
        self.rho[:, 0]  = self.e[:, 0] * 2 / 3 * self.mu * self.m_u / (self.k * self.T[:, 0])
        self.rho[:, -1] = self.e[:, -1] * 2 / 3 * self.mu * self.m_u / (self.k * self.T[:, -1])



    def central_x(self, var):

        """
        central difference scheme in x-direction
        """
        phi = var
        return np.roll(phi, 1, axis = 1) - np.roll(phi, -1, axis = 1) / (2 * self.dx)

    def central_y(self, var):

        """
        central difference scheme in y-direction
        """
        phi = var
        return np.roll(phi, 1, axis = 0) - np.roll(phi, -1, axis = 0) / (2 * self.dy)

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

#if __name__ == '__main__':
    # Run your code here