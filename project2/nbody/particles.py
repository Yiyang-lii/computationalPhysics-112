import numpy as np
import matplotlib.pyplot as plt


class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self, N):
        self.nparticles = N
        self._masses = np.ones((N, 1))
        self._positions = np.zeros((N, 3))
        self._velocities = np.zeros((N, 3))
        self._accelerations = np.zeros((N, 3))
        self._tags = np.linspace(1, N, N)
        self._time = 0.0

    @property
    def masses(self):
        return self._masses
    
    @masses.setter
    def masses(self, masses):
        if masses.shape[0] != self.nparticles:
            raise ValueError('Number of masses should be equal to number of particles')
        self._masses = masses
        return

    @property
    def positions(self):
        return self._positions
    
    @positions.setter
    def positions(self, positions):
        if positions.shape[0] != self.nparticles:
            raise ValueError('Number of positions should be equal to number of particles')
        self._positions = positions
        return
    
    @property
    def velocities(self):
        return self._velocities
    
    @velocities.setter
    def velocities(self, velocities):
        if velocities.shape[0] != self.nparticles:
            raise ValueError('Number of velocities should be equal to number of particles')
        self._velocities = velocities
        return

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        if tags.shape[0] != self.nparticles:
            raise ValueError('Number of tags should be equal to number of particles')
        self._tags = tags
        return
    
    def set_particles(self, pos, vel, acc, t):
        """
        Set particle properties
        """
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc
        self._time = t
        return

    def output(self, filename):
        """
        Output particle properties to a text file
        """
        header = "# time,tag,mass,x,y,z,vx,vy,vz,ax,ay,az\n"
        header+= "# s,,kg,m,m,m,m/s,m/s,m/s,m/s^2,m/s^2,m/s^2\n"

        np.savetxt(filename, 
                   np.hstack((np.ones((self.nparticles,1)) * self._time, 
                                       self._tags.reshape(-1,1), 
                                       self._masses, 
                                       self._positions, 
                                       self._velocities, 
                                       self._accelerations)), 
                                       delimiter=",", header=header, comments="")

        return
    
    def draw(self, dim=2, save=False, filename="particles.png"):
        """
        Draw particle positions
        """
        if dim == 2:
            fig = plt.figure()
            plt.scatter(self.positions[:,0], self.positions[:,1])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Particle projection in x-y plane, t = " + str(np.round(self._time)), fontsize=16)
            # set aspect ratio to be equal
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()
            return
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2])
            # set aspect ratio to be equal
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.tight_layout()
            plt.show()
            return
        else:
            raise ValueError("Invalid dimension")


if __name__ == '__main__':
    
    nparticles = 10
    dt = 0.1
    steps = 20
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

    # particles.draw(dim=3)
