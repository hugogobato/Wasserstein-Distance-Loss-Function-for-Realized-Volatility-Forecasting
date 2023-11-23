!pip install neuralforecast

!pip install ripser

from ripser import Rips
import persim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from typing import Optional, Union, Tuple


import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions import Bernoulli, Normal, StudentT, Poisson, NegativeBinomial

from torch.distributions import constraints


from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic
from neuralforecast.losses.numpy import rmse, mape

def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div

def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))

# %% ../../nbs/losses.pytorch.ipynb 7
class BasePointLoss(torch.nn.Module):
    """
    Base class for point loss functions.

    **Parameters:**<br>
    `horizon_weight`: Tensor of size h, weight for each timestamp of the forecasting window. <br>
    `outputsize_multiplier`: Multiplier for the output size. <br>
    `output_names`: Names of the outputs. <br>
    """

    def __init__(self, horizon_weight, outputsize_multiplier, output_names):
        super(BasePointLoss, self).__init__()
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight
        self.outputsize_multiplier = outputsize_multiplier
        self.output_names = output_names
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def _compute_weights(self, y, mask):
        """
        Compute final weights for each datapoint (based on all weights and all masks)
        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.
        """
        if mask is None:
            mask = torch.ones_like(y).to(y.device)

        if self.horizon_weight is None:
            self.horizon_weight = torch.ones(mask.shape[-1])
        else:
            assert mask.shape[-1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"

        weights = self.horizon_weight.clone()
        weights = torch.ones_like(mask, device=mask.device) * weights.to(mask.device)
        return weights * mask

class MSE_2DWD(BasePointLoss):
    from ripser import Rips
    import persim

    def __init__(self, horizon_weight=None):
        super(MSE_2DWD, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies datapoints to consider in loss.<br>

        **Returns:**<br>
        `mse`: tensor (single value).
        """
        rips = Rips(maxdim = 2, verbose=False)
        n=2
        wasserstein_dists = np.zeros((n))

        for i in range(2):
          dgm1 = rips.fit_transform(y_hat[:,int(y_hat.shape[1]/2*i):int(y_hat.shape[1]/2*(i+1))].detach().numpy())
          dgm2 = rips.fit_transform(y[:,int(y_hat.shape[1]/2*i):int(y_hat.shape[1]/2*(i+1))].detach().numpy())
          wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)

        losses = torch.sum((y - y_hat) ** 2)/(y_hat.shape[0]*y_hat.shape[1]) + torch.tensor((np.sum(wasserstein_dists))/n)
        return losses