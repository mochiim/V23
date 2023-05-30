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
        self.T = np.zeros([self.Ny, self.Nx])       # temperature
        self.P = np.zeros([self.Ny, self.Nx])       # pressure
        self.rho = np.zeros([self.Ny, self.Nx])     # density
        self.u = np.zeros([self.Ny, self.Nx])       # horizontal velocity
        self.w = np.zeros([self.Ny, self.Nx])       # vertical velocity
        self.e = np.zeros([self.Ny, self.Nx])       # internal energy
        self.gauss = np.zeros([self.Ny, self.Nx])   # gaussian distribution

    def initialise(self, Gauss = False):

        """
        initialise temperature, pressure, density and internal energy
        """
        nabla = 2.5 + .01

        # 2D Gaussian which is added to initial temperature if turned on
        if Gauss:
            for i in range(self.Ny - 1):
                for j in range(self.Nx - 1):
                    y, x = i, j
                    A = .1 * self.T_sun
                    mean_x = 150; mean_y = 80
                    sigma = 20
                    gauss = A * np.exp( (-.5 * ((x - mean_x) / sigma) ** 2) + (.5 * ((y - mean_y) / sigma)) **2 )
                    self.gauss[i, j] = gauss
            print("Gaussian perturbation: on")

        else:
            print("Gaussian perturbation: off")

        # initial values for temperature and pressure is given in lecture notes
        self.T[-1, :] = self.T_sun + self.gauss[-1, 0] 
        self.P[-1, :] = self.P_sun
        # internal energy and density is calculated from eos for an ideal gas
        self.e[-1, :] = 3/2 * self.P_sun 
        self.rho[-1, :] = 2/3 * self.P_sun * self.mu * self.m_u / (self.k * self.T_sun)

        for i in range(self.Ny - 1, 0, -1):
            dM = 4 * np.pi * self.R_sun ** 2 * self.rho[i, :]
            dP = - self.g * self.rho[i, :]
            dT = nabla * self.T[i, :] / self.P[i, :] * dP

            self.T[i - 1, :] = self.T[i, :] - dT * self.dy + self.gauss[i, :]
            self.P[i - 1, :] = self.P[i, :] - dP * self.dy
            self.e[i - 1, :] = 3 / 2 * self.P[i - 1, :]
            self.rho[i - 1, :] = 2/3 * self.e[i-1, :] * self.mu * self.m_u / (self.k * self.T[i-1, :])

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
        self.rho_dt  = - rho * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(rho, u) - w * self.upwind_y(rho, w)

        # horizontal component of momentum equation
        self.drhou_dt = rho * u * (self.upwind_x(u, u) + self.upwind_y(w, u)) - u * self.upwind_x(rho * u, u) - w * self.upwind_y(rho * u, w) - self.central_x(P)

        # vertical component of momentum equation
        self.drhow_dt = rho * w * (self.upwind_y(w, w) + self.upwind_x(u, w)) - w * self.upwind_y(rho * w, w) - u * self.upwind_x(rho * w, u) - self.central_y(P) + rho * self.g

        # energy equation
        self.de_dt = - e * (self.central_x(u) + self.central_y(w)) - u * self.upwind_x(e, u) - w * self.upwind_y(e, w) - P * (self.central_x(u) + self.central_y(w))


        # compute relative change for different variables, making sure avoid division by 0
        rel_rho = np.abs( self.rho_dt / rho )
        rel_rhou = np.abs( self.drhou_dt / (rho * u * ma.masked_where(u != 0, u)) )
        rel_rhow = np.abs( self.drhow_dt /  (rho * w * ma.masked_where(w != 0, w)) )
        rel_x = np.abs( u / self.dx )
        rel_y = np.abs( w / self.dy )
        rel_e = np.abs( self.de_dt / e )

        d = max([np.max(rel_rho), np.max(rel_rhou), np.max(rel_rhow), np.max(rel_x), np.max(rel_y), np.max(rel_e)])

        # in case all relative values are 0
        if d == 0:
            d = 1
            self.dt = p / d

        # avoiding too small time steps
        if d < 0.01:
            d = 0.01

            self.dt = p / d

        return self.dt

    def boundary_conditions(self):

        """
        boundary conditions for energy, density and velocity
        """
        # vertical boundary: vertical velocity
        self.w[0, :] = 0
        self.w[-1, :] = 0

        # vertical boundary: horizontal velocity
        self.u[0, :] = ( - self.u[2, :] + 4 * self.u[1, :] ) / 3
        self.u[-1, :] = ( - self.u[-2, :] + 4 * self.u[-1, :] ) / 3

        # vertical boundary: internal energy
        self.e[0, :]    = ( - self.e[2, :] + 4 * self.e[1, :] ) / ( 3 - 2 * self.mu * self.m_u * self.g / ( self.k * self.T[0, :] ) * self.dy )
        self.e[-1, :]   = ( - self.e[-3, :] + 4 * self.e[-2, :] ) / ( 3 + 2 * self.mu * self.m_u * self.g / ( self.k * self.T[-1, :] ) * self.dy )

        # vertical boundary: density
        self.rho[0, :]  = self.e[0, :] * 2 / 3 * self.mu * self.m_u / (self.k * self.T[0, :])
        self.rho[-1, :] = self.e[-1, :] * 2 / 3 * self.mu * self.m_u / (self.k * self.T[-1, :])

    def central_x(self, var):

        """
        central difference scheme in x-direction
        """
        phi = var
        phi_before = np.roll(phi, -1, axis = 1)
        phi_after = np.roll(phi, 1, axis = 1)

        dphi = phi_after - phi_before / (2 * self.dx)

        return dphi

    def central_y(self, var):

        """
        central difference scheme in y-direction
        """
        phi = var
        phi_before = np.roll(phi, -1, axis = 0)
        phi_after = np.roll(phi, 1, axis = 0)

        dphi = phi_after - phi_before / (2 * self.dy)

        return dphi

    def upwind_x(self, var, vel):

        """
        upwind difference scheme in x-direction
        """
        phi = var
        u = vel

        phi_before = np.roll(phi, -1, axis = 1)
        phi_after = np.roll(phi, 1, axis = -1)

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

        phi_before = np.roll(phi, -1, axis = 0)
        phi_after = np.roll(phi, 1, axis = 0)

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
        self.u[:] = (self.rho * self.u + self.drhou_dt * self.dt) / self.rho
        self.w[:] = (self.rho * self.w + self.drhow_dt * self.dt) / self.rho
        self.e[:] = self.e + self.de_dt * self.dt

        self.boundary_conditions()

        self.P[:] = 2 / 3 * self.e
        self.T[:] = self.P * self.m_u * self.mu / (self.rho * self.k)

        return self.dt

if __name__ == '__main__':
    vis = FVis.FluidVisualiser()
    test = Hydrodynamics()
    test.initialise(Gauss = False)
    test.hydro_solver()

    #vis.save_data(200, test.hydro_solver, rho = test.rho, u = test.u, w = test.w, e = test.e, P = test.P, T = test.T)
    # Folder: FVis_output_2023-05-28_13-16

    #vis.animate_2D("T", folder = "FVis_output_2023-05-28_13-16")
    
