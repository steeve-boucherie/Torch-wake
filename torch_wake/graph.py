"""Prototype of Graph NN for powe predictions."""
import logging
import warnings
# from itertools import product
# import os

import numpy as np

# from pandas import DataFrame

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import relu

from torch_geometric.nn import GCNConv


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def create_graph(
    x: np.ndarray,
    y: np.ndarray,
    max_dist: float = 1000
) -> Tensor:
    """
    Given the coordinates of the turbines, create the corresponding \
        graph by connecting all turbines within the distance threshold.

    Parameters
    ----------
    x: np.ndarray
        Input array containing the x-coordinates (Easting) of the \
        the turbines.
    y: np.ndarray
        Input array containing the y-coordinates (Northng) of the \
        the turbines.
    max_dist: float
        A number defining the maximum distance for which turbine \
        can form an edge of the graph.

    Returns
    -------
        Tensor

    Raises
    ------
        ValueError
    """
    # Internal
    def _verify_shape(arr: np.ndarray, ax: str):
        if arr.ndim == 1:
            msg = 'Expect 1-dimensional array for the coordinate. ' \
                  f'Array shape for array {ax}-coordinate: {arr.shape}.\n' \
                  'Flattening array.'
            logger.warning(msg)
            warnings.warn(msg)

        return arr.flatten()

    # Check inputs
    x = _verify_shape(x, 'x')
    y = _verify_shape(y, 'y')

    if not len(x) == len(y):
        msg = 'Arrays of x- and y-coordinates must have the same length.\n' \
              f'Length of x-coordnate: {len(x)}\nLength of y-coordnate: {len(y)}.\n' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    # Build graphs
    n_turbines = len(x)
    edges = []
    for i in range(n_turbines):
        for j in range(i + 1, n_turbines):
            d = float((x[i]**2 + y[j]**2)**.5)
            edges += [[], [(i, j)]][d <= max_dist]
            edges += [[], [(j, i)]][d <= max_dist]

    if len(edges) == 0:
        msg = 'Could not create any edges. Adjust distance threshold ' \
              'or check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    return torch.tensor(edges).t().contiguous()


# NN
class TurbineGNN(nn.Module):
    """Graph Neural Network for turbine power prediction."""

    def __init__(
        self,
        input_features: int,
        hidden_dim: int = 64
    ):
        """
        Initialize class instance.

        Parameters
        ----------
        input_features: int
            The number of feature in the input.
        hidden_dim: int
            Number of dimension of the hidden layers.
        """
        super(TurbineGNN, self).__init__()
        self.layers = [
            GCNConv(input_features, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, 32),
        ]
        self.predictor = nn.Linear(32, 1)
        self.dropout = nn.Dropout(.1)

    def forward(self, data):
        """Implement forward method"""
        # Get data
        x, edge_index = data.x, data.edge_index

        # Run Graph convolution
        for layer in self.layers:
            x = layer(x, edge_index)
            x = relu(x)
            x = self.dropout(x)

        # Make the prediction
        x = self.predictor(x)

        return x
