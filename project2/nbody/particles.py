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

    def set_initial_conditions(self, positions, velocities):
        self._positions = positions
        self._velocities = velocities

    def force(self, i, j, G=0.001, r_softening=0.01):
        r = self._positions[j] - self._positions[i]
        return G * self._masses[i] * self._masses[j] * r / (r + r_softening)**3
    
    def cal_acceleration(self, G=0.001, r_softening=0.01):
        for i in range(self.nparticles):
            for j in range(self.nparticles):
                if j > i:
                    self._accelerations[i] += self.force(i, j, G, r_softening) / self._masses[i]
                    self._accelerations[j] -= self.force(j, i, G, r_softening) / self._masses[j]
        return

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self._positions[:, 0], self._positions[:, 1], self._positions[:, 2])
        plt.show()

if __name__ == '__main__':
    p = Particles(2)
    p.set_masses(np.array([[1], [2]]))
    p.set_initial_conditions(np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0]]))
    p.plot()
