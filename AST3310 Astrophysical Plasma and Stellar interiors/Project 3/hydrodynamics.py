import FVis3 as FVis
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma


class Hydrodynamics:
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
        self.rho[:] = self.P * self.mu * self.m_u / ( self.k * self.T)
        self.u[:] = 0

    def timestep(self):

        """
        calculate timestep
        """
        p = .1
        u = self.u
        w = self.w
        P = self.P
        rho = self.rho
        e = self.e

        # continuity equation
        self.rho_dt  = - rho * (self._central_x(u) + self._central_y(w)) - u * self._upwind_x(rho, u) - w * self._upwind_y(rho, w)

        # horizontal component of momentum equation
        self.drhou_dt = rho * u * (self._upwind_x(u, u) + self._upwind_y(w, u)) - u * self._upwind_x(rho * u, u) - w * self._upwind_y(rho * u, w) - self._central_x(P)

        # vertical component of momentum equation
        self.drhow_dt = rho * w * (self._upwind_y(w, w) + self._upwind_x(u, w)) - w * self._upwind_y(rho * w, w) - u * self._upwind_x(rho * w, u) - self._central_y(P) + rho * self.g

        # energy equation
        self.de_dt = - e * (self._central_x(u) + self._central_y(w)) - u * self._upwind_x(e, u) - w * self._upwind_y(e, w) - P * (self._central_x(u) + self._central_y(w))

        # compute relative change for different variables
        rel_rho = max(abs( self.rho_dt / rho ))
        rel_rhou = max(abs( self.drhou_dt / (rho * u * self.masked_where(u != 0, self.u).mask) ))
        rel_rhow = max(abs( self.drhow_dt /  (rho * w * self.masked_where(w != 0, w).mask) ))
        rel_x = max(abs( u / self.dx ))
        rel_y = max(abs( w / self.dy ))
        rel_e = max(abs( self.de_dt / e ))

        d = max([rel_rho, rel_rhou, rel_rhow, rel_x, rel_y, rel_e])

        if d == 0:
            d = 1
            self.dt = p / d
        
        if d < 0.01:
            d = 0.01
            self.dt = p / d

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
        phi_before = np.roll(phi, -1, axis = 0)
        phi_after = np.roll(phi, 1, axis = 0)

        dphi = phi_after - phi_before / (2 * self.dx)

        return dphi

    def central_y(self, var):

        """
        central difference scheme in y-direction
        """
        phi = var
        phi_before = np.roll(phi, -1, axis = 1)
        phi_after = np.roll(phi, 1, axis = 1)

        dphi = phi_after - phi_before / (2 * self.dy)

        return dphi

    def upwind_x(self, var, vel):

        """
        upwind difference scheme in x-direction
        """
        phi = var
        u = vel
        
        phi_before = np.roll(phi, -1, axis = 0)
        phi_after = np.roll(phi, 1, axis = 0)

        # u >= 0
        pos_u = ma.masked_greater_equal(u, 0).mask * (phi - phi_before) / self.dx

        # u < 0
        neg_u = ma.masked_less(u, 0).mask * (phi_after - phi) / self.dx

        dphi = pos_u + neg_u

        return dphi
        
    def upwind_y(self, var, vel):

        """
        upwind difference scheme in y-direction
        """
        phi = var
        w = vel

        phi_before = np.roll(phi, -1, axis = 1)
        phi_after = np.roll(phi, 1, axis = 1)

        # w >= 0
        pos_w = ma.masked_greater_equal(w, 0).mask * (phi - phi_before) / self.dy

        # w < 0
        neg_w = ma.masked_less(w, 0).mask * (phi_after - phi) / self.dy

        dphi = pos_w + neg_w

        return dphi


    def hydro_solver(self):

        """
        hydrodynamic equations solver
        """
        self.timestep()

        self.rho[:] = self.rho + self.rho_dt + self.dt
        self.u[:] = (self.rho * self.u + self.rhou_dt * self.dt) / self.rho
        self.w[:] = (self.rho * self.w + self.rhow_dt * self.dt) / self.rho
        self.e[:] = e + self.e_dt * self.dt

        self.boundary_conditions()

        self.P[:] = 2 / 3 * self.e
        self.T[:] = self.P * self.m_u * self.mu / (self.rho * self.k)

        return self.dt

if __name__ == '__main__':
    test = Hydrodynamics()
    test.initialise()
    vis = FVis.FluidVisualiser()
    vis.animate_2D("T", save = False, folder = "FVis_output_test")