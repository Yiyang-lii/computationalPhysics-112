import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        # TODO
        self.particles = particles
        self.setup()

        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4" and "leapfrog"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        
        # TODO
        self._G = G
        self._rsoft = rsoft
        self._method = method
        self._io_freq = io_freq
        self._io_header = io_header
        self._io_screen = io_screen
        self._visualization = visualization
        self.time = 0

        return

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        # TODO
        time = self.time
        nsteps = int(tmax / dt)
        
        particles = self.particles

        if self._method == "Euler":
            self._advance_particles = self._advance_particles_Euler
        elif self._method == "RK2":
            self._advance_particles = self._advance_particles_RK2
        elif self._method == "RK4":
            self._advance_particles = self._advance_particles_RK4
        elif self._method == "leapfrog":
            self._advance_particles = self._advence_particles_leapfrog
        else:
            print("Error: method not supported")
            quit()

        for i in range(nsteps + 1):

            if (time + dt) >= tmax:
                dt = tmax - time
                time = tmax

            particles = self._advance_particles(dt, particles, time)
        
            folder = "data_" + self._io_header
            Path(folder).mkdir(parents=True, exist_ok=True)

            if (i % self._io_freq == 0):
                fn = self._io_header + "_" + str(i).zfill(6) + ".dat"
                fn = folder + "/" + fn 
                self.particles.output(fn)

                if self._visualization:
                    particles.draw(dim=2)
                
                if self._io_screen:
                    print("Time: ", time, "; Total time: ", tmax)
                    print("Step: ", i, "; Total steps: ", nsteps, end="\n\n")
                
            time += dt

        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        rsoft = self._rsoft
        G = self._G

        # TODO
        accelerations = _acceleration_kernal(nparticles, masses, positions, accelerations, rsoft, G)

        return accelerations
        
    def _advance_particles_Euler(self, dt, particles, time):

        #TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, mass, pos)

        pos += vel * dt
        vel += acc * dt
        acc = self._calculate_acceleration(nparticles, mass, pos)
        
        particles.set_particles(pos, vel, acc, time)
        
        return particles

    def _advance_particles_RK2(self, dt, particles, time):

        # TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, mass, pos)

        pos2 = pos + vel*dt
        vel2 = vel + acc*dt
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)

        pos2 = pos2 + vel2*dt
        vel2 = vel2 + acc2*dt

        pos = 0.5 * (pos + pos2)
        vel = 0.5 * (vel + vel2)
        acc = self._calculate_acceleration(nparticles, mass, pos)

        particles.set_particles(pos, vel, acc, time)

        return particles

    def _advance_particles_RK4(self, dt, particles, time):
        
        #TODO
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, mass, pos)

        dt2 = dt/2
        pos1 = pos + vel*dt2
        vel1 = vel + acc*dt2
        acc1 = self._calculate_acceleration(nparticles, mass, pos1)

        pos2 = pos + vel1*dt2
        vel2 = vel + acc1*dt2
        acc2 = self._calculate_acceleration(nparticles, mass, pos2)

        pos3 = pos + vel2*dt
        vel3 = vel + acc2*dt
        acc3 = self._calculate_acceleration(nparticles, mass, pos3)

        pos = pos + (vel + 2 * vel1 + 2 * vel2 + vel3) * dt / 6
        vel = vel + (acc + 2 * acc1 + 2 * acc2 + acc3) * dt / 6
        acc = self._calculate_acceleration(nparticles, mass, pos)

        particles.set_particles(pos, vel, acc, time)

        return particles
    
    def _advence_particles_leapfrog(self, dt, particles, time):
        
        nparticles = particles.nparticles
        mass = particles.masses

        pos = particles.positions
        vel = particles.velocities
        acc = self._calculate_acceleration(nparticles, mass, pos)

        vel += acc * dt / 2 # kick 1/2 steps forward
        pos += vel * dt # drift 1 step forward
        acc = self._calculate_acceleration(nparticles, mass, pos)

        vel += acc * dt / 2 # kick 1/2 steps forward

        particles.set_particles(pos, vel, acc, time)

        return particles
    
@njit(parallel=True)
def _acceleration_kernal(nparticles, mass, positions, accelerations, rsoft, G):
    """
    Calculate the acceleration of the particles
    """
    for i in prange(nparticles):
        for j in prange(nparticles):
            if j > i:
                rij = positions[i, :] - positions[j, :]
                r = np.sqrt(np.sum(rij**2 + rsoft**2))
                force = -G * mass[i, 0] * mass[j, 0] * rij / r**3
                accelerations[i, :] += force[:] / mass[i, 0]
                accelerations[j, :] -= force[:] / mass[j, 0]
    
    return accelerations
    

if __name__ == "__main__":
    # test 
    nparticles = 10
    dt = 0.1
    steps = 50
    tmax = 5
    G = 1.0
    rsoft = 0.01
    
    particles = Particles(nparticles)
    
    total_mass = 20
    particles.masses = np.ones((nparticles, 1))
    particles.masses = total_mass * particles.masses / nparticles
    
    np.random.seed(46)
    particles.positions = np.random.rand(nparticles, 3)
    particles.velocities = np.zeros((nparticles, 3))
    particles.tags = np.linspace(1, nparticles, nparticles)
   
    mean = np.mean(particles.positions, axis=0)
    std = np.std(particles.positions, axis=0)
    particles.positions = (particles.positions - mean) / std
    
    set_num_threads(4)

    simulation = NBodySimulator(particles)
    simulation.setup(G=G, rsoft=rsoft, method="RK4", io_freq=10, io_header="nbody", io_screen=True, visualization=False)
    simulation.evolve(dt=dt, tmax=tmax)