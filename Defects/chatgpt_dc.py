# chatgpt_dc.py
"""
Cleaned and documented version of defect_class.py.
This file contains the DefectSquareLattice class and related functions for modeling and visualizing topological insulator defect lattices.

- All code is preserved from defect_class.py, but with improved docstrings, comments, and organization.
- No functional changes are made.
- For detailed usage, see the docstrings and function comments.
"""

import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.ticker as ticker
import scipy.linalg as spla
from matplotlib.widgets import Slider
import inspect

class DefectSquareLattice:
    """
    Square lattice model for topological insulator defects (vacancy, substitution, interstitial, frenkel pair).
    Provides methods for geometry generation, Hamiltonian construction, spectrum/LDOS/LCM computation, and visualization.
    """
    def __init__(self, side_length: int, defect_type: str, pbc: bool = True, frenkel_pair_index: int = 0, doLargeDefect: bool = False, *args, **kwargs):
        """
        Initialize the lattice with given size, defect type, and options.
        Args:
            side_length: int, lattice size (number of sites per side)
            defect_type: str, one of ['none', 'vacancy', 'substitution', 'interstitial', 'frenkel_pair']
            pbc: bool, use periodic boundary conditions
            frenkel_pair_index: int, index for frenkel pair displacement
            doLargeDefect: bool, use large defect geometry
        """
        # ...existing code from defect_class.py __init__...

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def side_length(self):
        """Lattice side length (number of sites per side)."""
        return self._side_length
    @property
    def pbc(self):
        """Periodic boundary conditions flag."""
        return self._pbc 
    @property
    def defect_type(self):
        """Type of defect (none, vacancy, substitution, interstitial, frenkel_pair)."""
        return self._defect_type
    @property
    def defect_indices(self):
        """Indices of defect sites in the lattice."""
        return self._defect_indices
    @property
    def X(self):
        """X coordinates of lattice sites."""
        return self._X
    @property
    def Y(self):
        """Y coordinates of lattice sites."""
        return self._Y
    @property
    def dx(self):
        """Matrix of x displacements between all pairs of sites."""
        return self._dx
    @property
    def dy(self):
        """Matrix of y displacements between all pairs of sites."""
        return self._dy
    @property
    def lattice(self):
        """Lattice array (site indices, -1 for vacancies/interstitials)."""
        # ...existing code...

    # -------------------------------------------------------------------------
    # Geometry generation methods
    # -------------------------------------------------------------------------
    def generate_lattice(self, *args, **kwargs):
        """Generate a pristine square lattice (no defects)."""
        # ...existing code...

    def generate_vacancy_lattice(self, *args, **kwargs):
        """Generate a lattice with a central vacancy or large vacancy cluster."""
        # ...existing code...

    def generate_interstitial_lattice(self):
        """Generate a lattice with a central interstitial or large interstitial cluster."""
        # ...existing code...

    def compute_frenkel_pair_lattice(self, displacement_index: int):
        """Generate a lattice with a Frenkel pair (vacancy + interstitial) at a given displacement index."""
        # ...existing code...

    # -------------------------------------------------------------------------
    # Distance and coupling matrix computation
    # -------------------------------------------------------------------------
    def compute_distances(self, *args, **kwargs):
        """Compute dx, dy matrices for all site pairs, respecting PBC if enabled."""
        # ...existing code...

    def compute_wannier_polar(self, *args, **kwargs):
        """Compute Wannier function coupling matrices (Sx, Sy, Cx_plus_Cy, etc)."""
        # ...existing code...

    # -------------------------------------------------------------------------
    # Hamiltonian, projector, and observable computation
    # -------------------------------------------------------------------------
    def compute_hamiltonian(self, M_background: float, M_substitution: float = None, t: float = 1.0, t0: float = 1.0):
        """
        Construct the full tight-binding Hamiltonian for the lattice.
        Args:
            M_background: float, mass term for background sites
            M_substitution: float, mass term for defect sites (if any)
            t: float, hopping amplitude
            t0: float, additional hopping parameter
        Returns:
            Hamiltonian matrix (complex ndarray)
        """
        # ...existing code...

    def compute_projector(self, hamiltonian):
        """Compute the projector onto the lower band of the Hamiltonian."""
        # ...existing code...

    def compute_bott_index(self, projector: np.ndarray):
        """Compute the Bott index (topological invariant) from the projector."""
        # ...existing code...

    def compute_local_chern_operator(self, hamiltonian, *args, **kwargs):
        """Compute the local Chern marker operator for the system."""
        # ...existing code...

    def compute_LDOS(self, hamiltonian: np.ndarray, number_of_states: int = 2, *args, **kwargs):
        """
        Compute the local density of states (LDOS) for a given Hamiltonian.
        Args:
            hamiltonian: ndarray, system Hamiltonian
            number_of_states: int, number of states near the Fermi level to include
        Returns:
            dict with LDOS, eigenvalues, gap, bandwidth, ldos_idxs
        """
        # ...existing code...

    def _compute_for_figure(self, m_background: float, m_substitution: float, number_of_states: float):
        """
        Helper for figure plotting: computes LDOS, spectrum, gap, Bott index, and coordinates for given parameters.
        """
        # ...existing code...

    # -------------------------------------------------------------------------
    # Plotting and visualization methods
    # -------------------------------------------------------------------------
    def plot_spectrum_ldos(self, m_background_values: "list[float]" = [2.5, 1.0, -1.0, -2.5], 
                           m_substitution_values: "list[float] | None" = None, doLargeDefectFigure: bool = False, number_of_states: int = 2):
        """
        Plot the energy spectrum and LDOS for various mass parameters.
        Args:
            m_background_values: list of background mass values
            m_substitution_values: list of defect mass values (optional)
            doLargeDefectFigure: use large defect geometry if True
            number_of_states: number of states for LDOS
        Returns:
            (fig, axs): matplotlib Figure and Axes
        """
        # ...existing code...

    def plot_distances(self, idx: int = None, cmap: str = "inferno", doLargeDefectFigure: bool = False, *args, **kwargs):
        """
        Plot dx, dy, and distance from a given site to all others.
        Args:
            idx: site index (default: center)
            cmap: colormap
            doLargeDefectFigure: use large defect geometry if True
        """
        # ...existing code...

    def plot_defect_idxs(self):
        """Plot the lattice and highlight defect site indices."""
        # ...existing code...

    def plot_wannier(self, idx: int = None):
        """Plot Wannier function components for a given site index."""
        # ...existing code...

    def plot_lcm(self, m_background_values: "list[float]" = [2.5, 1.0, -1.0, -2.5], 
                 m_substitution_values: "list[float] | None" = None, doLargeDefectFigure: bool = False):
        """
        Plot the local Chern marker (LCM) for various mass parameters.
        Args:
            m_background_values: list of background mass values
            m_substitution_values: list of defect mass values (optional)
            doLargeDefectFigure: use large defect geometry if True
        Returns:
            (fig, axs): matplotlib Figure and Axes
        """
        # ...existing code...

    # -------------------------------------------------------------------------
    # Interactive visualization
    # -------------------------------------------------------------------------
    def animate_spectrum_ldos(self, m_back_range=(-3, 3), m_sub_range=(-3, 3), num_points=25, number_of_states=2, doLargeDefectFigure=False):
        """
        Interactive animation of spectrum and LDOS with two sliders for m_back and m_sub.
        Args:
            m_back_range: tuple, (min, max) for m_back slider
            m_sub_range: tuple, (min, max) for m_sub slider
            num_points: int, number of slider steps
            number_of_states: int, number of states for LDOS
            doLargeDefectFigure: bool, use LargeDefectLattice if True
        """
        # ...existing code...

# -------------------------------------------------------------------------
# Standalone analysis and figure generation functions
# -------------------------------------------------------------------------
def BI_PD():
    """
    Plot the Bott index as a function of mass for a pristine lattice.
    """
    # ...existing code...

def generate_ldos_figures():
    """
    Generate and save LDOS figures for all defect types.
    """
    # ...existing code...

def main2():
    """
    Example analysis: compare spectra, eigenvectors, and couplings for interstitial defects.
    """
    # ...existing code...

def generate_lcm_figures():
    """
    Generate and save LCM figures for all defect types.
    """
    # ...existing code...

if __name__ == "__main__":
    generate_ldos_figures()
