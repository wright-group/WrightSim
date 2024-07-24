import numpy as np
from ..mixed import propagate
from typing import List


class AbstractRKHamiltonian:
    """boilerplate for Runge Kutta propagator
    """

    # these need to be explicitly available to the propagator
    # each subclass needs to define these (e.g. declare in __init__)
    recorded_indices: List[int]
    rho: np.array
    omega: np.array  # density matrix frequencies, in rad/fs
    labels: List[str]
    tau: np.array  # relaxation rates, in fs

    def __init__(self):
        self._gamma_matrix = None
        self._omega_matrix = None
        self.propagator = propagate.runge_kutta

    def matrix(self, efields, time) -> np.ndarray:
        """Generate the time dependent Hamiltonian Coupling Matrix

        Parameters
        ----------
        efields : ndarray<Complex>
            Contains the time dependent electric fields.
            Shape (M x T) where M is number of electric fields, and T is number of timesteps.
        time : 1-D array <float64>
            The time step values
        """
        out = 0.5j * self.rabi_matrix(efields)
        out *= np.exp(1j * self.omega_matrix * time[None, None, :])
        out -= self.gamma_matrix
        return out.transpose(2, 0, 1)  # move time axis first

    @property
    def gamma_matrix(self):
        if self._gamma_matrix is None:
            self._gamma_matrix = np.zeros((self.omega.size,)*2)
            np.fill_diagonal(self._gamma_matrix, 1 / self.taus)
        return self._gamma_matrix[:, :, None]

    @property
    def omega_matrix(self):
        if self._omega_matrix is None:
            self._omega_matrix = self.omega[:, None] - self.omega[None, :]
        return self._omega_matrix[:, :, None]

    def rabi_matrix(self, efields:List[np.array]):
        """
        define the coupling matrix here--dipoles * efields
        return matrix of shape [to, from, time]
        Usage
        -----
            E1, E2, E3 = efields
            out = np.zeros((self.rho.size, self.rho.size, efields[0].size), dtype=complex)
            out[to_index, from_index] = -E1 * mu_fi
            return out
        """
        raise NotImplementedError

    @property
    def attrs(self) -> dict:
        """define quantities to send to output attrs dict"""
        return {
            "rho": self.rho,
            "omega": self.omega,
            "propagator": "runge_kutta",
            "taus": self.taus
        }

    def matshow(self, ax, efield:float=1.0, fontsize=10):
        """wrapper for matplotlib.pyplot.matshow to quickly show the matrix
        """
        mat = self.matrix([np.array([efield])] * 3, np.array([0])).squeeze()
        art = ax.matshow(np.abs(mat))
        ax.get_figure().colorbar(art, ax=ax, label="amplitude")
        labels = [f"{i}: {label}" for i, label in enumerate(self.labels)]
        ax.set_yticks(np.arange(mat.shape[0]), labels=labels, fontsize=fontsize)
        ax.set_xticks(np.arange(mat.shape[0]), labels=labels, rotation=45, ha="left", fontsize=fontsize, rotation_mode="anchor")
        ax.grid(c="w", ls=":")

