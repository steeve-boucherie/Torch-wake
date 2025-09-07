"""
Utility classes and method to generate synthetic data for training \
model.
"""
import logging
from typing import List, Optional

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pydantic import BaseModel, ConfigDict

from py_wake.site import UniformWeibullSite

from scipy import stats

import xarray as xr


# LOGGER
logger = logging.getLogger(__name__)


# Layout Generator
class LayoutGenerator(BaseException):
    """
    Generate regular grid layouts for wind farms.

    Attributes
    ----------
    spacing_x: float
        A number defining the spacing between turbines in x direction (meters)
    spacing_y: float
        A number defining the spacing between turbines in y direction (meters)
    """

    spacing_x: float = 800
    spacing_y: float = 800

    # def __init__(self, spacing_x: float = 800, spacing_y: float = 600):
    #     """
    #     Initialize layout generator.

    #     Args:
    #         spacing_x: Spacing between turbines in x direction (meters)
    #         spacing_y: Spacing between turbines in y direction (meters)
    #     """
    #     self.spacing_x = spacing_x
    #     self.spacing_y = spacing_y

    def generate_grid(self, n_turbines: int) -> pd.DataFrame:
        """
        Generate a regular grid layout.

        Args:
            n_turbines: Total number of turbines

        Returns:
            DataFrame with columns: turbine_id, x, y
        """
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_turbines)))
        n_rows = int(np.ceil(n_turbines / n_cols))

        turbines = []
        turbine_id = 0

        for row in range(n_rows):
            for col in range(n_cols):
                if turbine_id >= n_turbines:
                    break

                x = col * self.spacing_x
                y = row * self.spacing_y

                turbines.append({
                    'turbine_id': turbine_id,
                    'x': x,
                    'y': y
                })
                turbine_id += 1

        return pd.DataFrame(turbines)

    def visualize_layout(self, layout_df: pd.DataFrame):
        """Visualize the wind farm layout."""
        plt.figure(figsize=(10, 8))
        plt.scatter(layout_df['x'], layout_df['y'], s=100, alpha=0.7)

        for _, row in layout_df.iterrows():
            plt.annotate(
                f"T{int(row['turbine_id'])}",
                (row['x'], row['y']),
                xytext=(5, 5),
                textcoords='offset points'
            )

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Wind Farm Layout')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


# Wind Condition Generator
class MixtureFlowGenerator(BaseModel):
    """
    A class to generate wind conditions using a mixture of Von-Mises \
        distribution for the wind direction.

    Attributes
    ----------
    n_sector: int
        The number of wind sector to consider (default = 12).
    n_comps: int
        The number of components of the mixture model.
    weib_a: float
        A number defining the average weibull scale parameter. \
        The scale parameter of each sector is sampled from a normal \
        distribution center on weib_a with unit standard deviation.
    weib_k: float
        A number defining the shape parameter of the weibull distributions. \
        Uses a common shape for all sectors.
    ti_level: float
        A number defining the turbulence intensity level for the \
        generated time_series (constant for all timestamps)
    random_seed: int | None
        (Optional) Random seed number to ensure repeatability of the \
        results.
    """

    # Settings for the wind direction
    n_sector: int = 12
    n_comps: int = 4

    # Settings for the wind speed
    weib_a: float = 9.1
    weib_k: float = 2.1
    ti_level: float = 0.15

    # Options
    random_seed: Optional[int] = None

    # Internal
    comp_weights: List[float] = None
    wd_components: List[stats.vonmises] = None
    weibull_scales: List[float] = None

    # Class settings
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Properties
    @property
    def rng(self) -> np.random.RandomState:
        """Return the random Generator"""
        return np.random.RandomState(self.random_seed)

    # Methods
    def _is_init(self) -> None:
        """Test if the model is initialized."""
        elements = [
            self.comp_weights,
            self.wd_components,
            self.weibull_scales
        ]

        tests = np.array([
            elem is None for elem in elements
        ])

        if tests.any():
            msg = 'Model is not properly initialized.'
            logger.error(msg)
            raise ValueError(msg)

    def _init_wd_dist(self) -> None:
        """Init the components of the wind direction distributions."""
        # Get the components weights
        weights = stats.dirichlet.rvs(
            [1 for _ in range(self.n_comps)],
            random_state=self.rng
        )
        self.comp_weights = weights.ravel()

        # Get the locations and kappas
        locations = np.deg2rad([
            n * 360 / self.n_comps for n in range(self.n_comps)
        ])
        kappas = (
            stats.uniform(scale=2)
            .rvs(
                self.n_comps,
                random_state=self.rng
            )
        )

        self.wd_components = [
            stats.vonmises(loc=loc, kappa=kappa)
            for loc, kappa in zip(locations, kappas)
        ]

    def _init_weibull_scales(self) -> None:
        """Init the values of the scale parameters of the weibulls."""
        scales = stats.norm(loc=self.weib_a, scale=1).rvs(
            self.n_sector,
            random_state=self.rng
        )

        self.weibull_scales = scales.astype(float)

    def init_generator(self) -> 'MixtureFlowGenerator':
        """Initialize generator and set internal terms values."""
        logger.info('Initialize generator terms')
        self._init_wd_dist()
        self._init_weibull_scales()

        return self

    def sector_frequencies(self) -> List[float]:
        """
        Compute the sector frequencies.

        Returns
        -------
            List[float]
        """
        # FIXME: Make sector centered around 0
        bnds = np.deg2rad(np.linspace(0, 360, self.n_sector + 1))

        cdfs = [
            [
                w * (comp.cdf(bnds[n + 1]) - comp.cdf(bnds[n]))
                for n in range(self.n_sector)
            ]
            for w, comp in zip(self.comp_weights, self.wd_components)
        ]

        cdfs = np.array(cdfs).sum(axis=0).ravel()

        return cdfs

    def sample(self, n_samp: int) -> xr.Dataset:
        """
        Sample timeseries.

        Parameters
        ----------
        n_samp: int
            The number of samples

        Returns
        -------
            xr.Dataset
        """
        # Sample the sector and get the wd
        sectors = self.rng.choice(
            self.n_sector,
            size=n_samp,
            p=self.sector_frequencies()
        )
        width = 360 / self.n_sector
        wd = sectors * width + width / 2
        noise = (
            stats.uniform(loc=(-width / 2), scale=width)
            .rvs(n_samp, self.rng)
        )

        wd = wd + noise

        # Get the corresponding scales
        scales = np.take(self.weibull_scales, sectors)
        ws = (
            stats.weibull_min(self.weib_k, scale=scales)
            .rvs(n_samp, self.rng)
        )

        out = xr.Dataset(
            data_vars={
                'ws': ('time', ws),
                'wd': ('time', wd),
                'ti': ('time', np.ones_like(ws) * self.ti_level)
            },
            coords={
                'time': ('time', np.arange(n_samp))
            }
        )

        return out

    def create_site(self) -> UniformWeibullSite:
        """
        Initialize the pyWake's site object using the \
            defined wind conditions

        Returns
        -------
            UniformWeibullSite
        """
        site = UniformWeibullSite(
            p_wd=list(self.sector_frequencies()),
            a=list(self.weibull_scales),
            k=[self.weib_k for _ in range(self.n_sector)]
        )

        return site
