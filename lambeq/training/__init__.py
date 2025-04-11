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

__all__ = ['Checkpoint',
           'Dataset',

           'Model',
           'NumpyModel',
           'PennyLaneModel',
           'PytorchModel',
           'PytorchQuantumModel',
           'QuantumModel',
           'TketModel',

           'Optimizer',
           'NelderMeadOptimizer',
           'RotosolveOptimizer',
           'SPSAOptimizer',

           'Trainer',
           'PytorchTrainer',
           'QuantumTrainer',

           'TnPathOptimizer',
           'CachedTnPathOptimizer',

           'BinaryCrossEntropyLoss',
           'CrossEntropyLoss',
           'LossFunction',
           'MSELoss', ]

from lambeq.training.checkpoint import Checkpoint

from lambeq.training.dataset import Dataset

from lambeq.training.model import Model
from lambeq.training.numpy_model import NumpyModel
from lambeq.training.pennylane_model import PennyLaneModel
from lambeq.training.pytorch_model import PytorchModel
from lambeq.training.pytorch_quantum_model import PytorchQuantumModel
from lambeq.training.quantum_model import QuantumModel
from lambeq.training.tket_model import TketModel

from lambeq.training.optimizer import Optimizer
from lambeq.training.nelder_mead_optimizer import NelderMeadOptimizer
from lambeq.training.rotosolve_optimizer import RotosolveOptimizer
from lambeq.training.spsa_optimizer import SPSAOptimizer

from lambeq.training.trainer import Trainer
from lambeq.training.pytorch_trainer import PytorchTrainer
from lambeq.training.quantum_trainer import QuantumTrainer
from lambeq.training.tn_path_optimizer import (CachedTnPathOptimizer,
                                               TnPathOptimizer)

from lambeq.training.loss import (BinaryCrossEntropyLoss,
                                  CrossEntropyLoss,
                                  LossFunction,
                                  MSELoss)
