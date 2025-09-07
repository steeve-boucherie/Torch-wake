"""
Utility classes and method to generate synthetic data for training \
model.
"""
import logging
import warnings
from itertools import product
from typing import List, Optional
# from typing_extensions import Annotated

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import RandomState

import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator
)

from py_wake.site import UniformWeibullSite
from py_wake.wind_farm_models.wind_farm_model import SimulationResult

from sklearn.preprocessing import MinMaxScaler

from scipy import stats

import torch
from torch import Tensor

from torch_geometric.data import Data

import xarray as xr


# PACKAGE IMPORTS
from torch_wake.utils import drop_nondim_coords


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def _prepare_time_data(
    sim_res: SimulationResult,
    edge_index: Tensor,
    toi: int
) -> List[Data]:
    """
    Given the pywake time simulation results, prepare the data for training.

    Parameters
    ----------
    sim_res: SimulationResult
        An instance of pyWake's SimulationResult class to be used for generating \
        the training data.
    edge_index: Tensor
        An instance of torch.Tensor defining the edge connections.
    toi: int
        The ID of the turbine of interest - considered as the feature.

    Returns
    -------
        List[Data]
    """
    sim_res = (
        sim_res
        .rename({'Power': 'power'})
        .assign({
            'wd_x': np.cos(np.deg2rad(sim_res['wd'])),
            'wd_y': np.sin(np.deg2rad(sim_res['wd'])),
        })
    )

    groups = (
        xr.merge([
            drop_nondim_coords(sim_res[key])
            for key in ['x', 'y', 'ws', 'wd_x', 'wd_y', 'power']
        ])
        .pipe(lambda ds: (
            ds.assign(
                power_feat=xr.where(
                    ds['wt'] == 'toi',
                    0,
                    ds['power']
                )
            )
        ))
        .to_dataframe()
        .pipe(lambda df: (
            df.assign(**{
                c: (MinMaxScaler().fit_transform(df[[c]]))
                for c in df.columns
            })
        ))
        .astype(np.float32)        
        .reorder_levels(['time', 'wt'])
        .sort_index(axis=0)
        .reset_index()
        .groupby('time')
    )

    data_list = []
    all_keys = list(groups.groups.keys())
    for n, dt in enumerate(all_keys):
        if n == 0 or ((n + 1) % 1000 == 0):
            msg = f'Processing timestamp {n + 1:5d} of {len(all_keys):5d}'
            logger.info(msg)

        df = groups.get_group(dt)
        x = torch.tensor(df[['x', 'y', 'ws', 'wd_x', 'wd_y', 'power_feat']].values)
        y = torch.tensor(df['power'].values)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.toi = toi
        data_list.append(data)

    return data_list


def prepare_data(
    sim_res: SimulationResult,
    edge_index: Tensor,
    toi: int
) -> List[Data]:
    """
    Given the pywake simulation results, prepare the data for training.

    Parameters
    ----------
    sim_res: SimulationResult
        An instance of pyWake's SimulationResult class to be used for generating \
        the training data.
    edge_index: Tensor
        An instance of torch.Tensor defining the edge connections.
    toi: int
        The ID of the turbine of interest - considered as the feature.

    Returns
    -------
        List[Data]
    """
    if 'time' in sim_res.dims:
        data_list = _prepare_time_data(sim_res, edge_index, toi)

    else:
        msg = 'Only support simulation results with time, so far.'
        logger.err(msg)
        raise NotImplementedError(msg)

    return data_list


def dummy_wind(
    ws_range: Optional[np.ndarray] = None,
    wd_range: Optional[np.ndarray] = None,
    ti: float = .1
) -> xr.Dataset:
    """
    Generate dataset of dummy conditions using the product of \
        wind speed and wind direction arrays.

    Parameters
    ----------
    ws_range: np.ndarray | None
        (Optional) Array of wind speeds.
    wd_range: np.ndarray | None
        (Optional) Array of wind direction.
    ti: float
        A number defining the selected turbulence intensity level.

    Returns
    -------
        Dataset
    """
    # Get defaults
    ws_range = [ws_range, np.arange(3, 21)][ws_range is None]
    wd_range = [wd_range, np.arange(0, 360, 5)][wd_range is None]

    data = [[ws, wd] for ws, wd in product(ws_range, wd_range)]
    out = (
        pd.DataFrame(
            data=data,
            columns=['ws', 'wd'],
            index=pd.Index(
                data=np.arange(len(data)),
                name='time'
            )
        )
        .assign(ti=ti)
        .to_xarray()
    )

    return out


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
    rng: Optional[RandomState] = Field(default=None, validate_default=True)
    wd_components: List[stats.vonmises] = None
    weibull_scales: List[float] = None

    # Class settings
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Properties
    # @property
    # def rng(self) -> np.random.RandomState:
    #     """Return the random Generator"""
    #     return np.random.RandomState(self.random_seed)

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

    # Validator
    @field_validator('rng')
    @classmethod
    def init_rng(
        cls,
        v: Optional[RandomState],
        info: ValidationInfo,
    ) -> RandomState:
        """
        Initialize the random State by the selected seed number.

        Returns
        -------
            RandomState
        """
        random_seed = info.data['random_seed']
        
        if v is not None:
            msg = 'Improper initialization of random state. ' \
                  'This is an internal attribue. Use "random_seed"' \
                  'to control the selected seed.\n' \
                  f'Currently selected seed: {random_seed}.'  
            logger.warning(msg)
            warnings.warn(msg)

        return RandomState(random_seed)
