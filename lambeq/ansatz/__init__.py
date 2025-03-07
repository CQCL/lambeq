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

__all__ = ['BaseAnsatz', 'CircuitAnsatz', 'IQPAnsatz', 'MPSAnsatz',
           'Sim14Ansatz', 'Sim15Ansatz', 'Sim4Ansatz', 'Sim9Ansatz',
           'Sim9CxAnsatz', 'SpiderAnsatz', 'StronglyEntanglingAnsatz',
           'TensorAnsatz']

from lambeq.ansatz.base import BaseAnsatz
from lambeq.ansatz.circuit import (CircuitAnsatz, IQPAnsatz, Sim9Ansatz,
                                   Sim9CxAnsatz, Sim14Ansatz, Sim15Ansatz,
                                   Sim4Ansatz, StronglyEntanglingAnsatz)
from lambeq.ansatz.tensor import MPSAnsatz, SpiderAnsatz, TensorAnsatz
