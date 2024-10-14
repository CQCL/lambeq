# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Loss Functions
==============
Module containing loss functions to train lambeq's quantum models.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import numpy as jnp
    from types import ModuleType


class LossFunction(ABC):
    """Loss function base class.

    Attributes
    ----------
    backend : ModuleType
        The module to use for array numerical functions.
         Either numpy or jax.numpy.

    """

    def __init__(self, use_jax: bool = False) -> None:
        """Initialise a loss function.

        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy as `backend`.

        """

        self.backend: ModuleType

        if use_jax:
            from jax import numpy as jnp
            self.backend = jnp
        else:
            self.backend = np

    def _match_shapes(self,
                      y1: np.ndarray | jnp.ndarray,
                      y2: np.ndarray | jnp.ndarray) -> None:
        if y1.shape != y2.shape:
            raise ValueError('Provided arrays must be of equal shape. Got '
                             f'arrays of shape {y1.shape} and {y2.shape}.')

    def _smooth_and_normalise(self,
                              y: np.ndarray | jnp.ndarray,
                              epsilon: float
                              ) -> np.ndarray | jnp.ndarray:

        y_smoothed = y + epsilon

        l1_norms: np.ndarray | jnp.ndarray = self.backend.linalg.norm(
                                                y_smoothed,
                                                ord=1,
                                                axis=1,
                                                keepdims=True)

        return y_smoothed / l1_norms

    @abstractmethod
    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of loss function."""

    def __call__(self,
                 y_pred: np.ndarray | jnp.ndarray,
                 y_true: np.ndarray | jnp.ndarray) -> float:
        return self.calculate_loss(y_pred, y_true)


class CrossEntropyLoss(LossFunction):
    """Multiclass cross-entropy loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. Expected to be of shape
        [batch_size, n_classes], where each row is a probability
        distribution.
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. Expected to be of shape
        [batch_size, n_classes], where each row is a one-hot vector.

    """

    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.

        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._epsilon = epsilon

        super().__init__(use_jax)

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        self._match_shapes(y_pred, y_true)

        y_pred_smoothed = self._smooth_and_normalise(y_pred, self._epsilon)

        entropies = y_true * self.backend.log(y_pred_smoothed)
        loss_val: float = -self.backend.sum(entropies) / len(y_true)

        return loss_val


class BinaryCrossEntropyLoss(CrossEntropyLoss):
    """Binary cross-entropy loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. When `sparse` is `False`,
        expected to be of shape [batch_size, 2], where each row is a
        probability distribution. When `sparse` is `True`, expected to
        be of shape [batch_size, ] where each element indicates P(1).
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. When `sparse` is `False`, expected
        to be of shape [batch_size, 2], where each row is a one-hot
        vector. When `sparse` is `True`, expected to be of shape
        [batch_size, ] where each element is an integer indicating
        class label.

    """

    def __init__(self,
                 sparse: bool = False,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a binary cross-entropy loss function.

        Parameters
        ----------
        sparse : bool, default: False
            If True, each input element indicates P(1), else the
             probability distribution over classes is expected.
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._sparse = sparse
        super().__init__(use_jax, epsilon)

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of BCE loss function."""

        if self._sparse:
            # For numerical stability, it is convenient to reshape the
            #  sparse input to a dense representation.

            self._match_shapes(y_pred, y_true)

            y_pred_dense = self.backend.stack((1 - y_pred, y_pred)).T
            y_true_dense = self.backend.stack((1 - y_true, y_true)).T

            return super().calculate_loss(y_pred_dense, y_true_dense)
        else:
            return super().calculate_loss(y_pred, y_true)


class MSELoss(LossFunction):
    """Mean squared error loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted values from model. Shape must match y_true.
    y_true: np.ndarray or jnp.ndarray
        Ground truth values.

    """

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of MSE loss function."""

        self._match_shapes(y_pred, y_true)

        return float(self.backend.mean((y_pred - y_true) ** 2))
