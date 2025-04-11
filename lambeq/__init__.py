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

__all__ = [
        '__version__',
        '__version_info__',

        'ansatz',
        'core',
        'rewrite',
        'text2diagram',
        'tokeniser',
        'training',

        'BaseAnsatz',
        'CircuitAnsatz',
        'IQPAnsatz',
        'MPSAnsatz',
        'Sim14Ansatz',
        'Sim15Ansatz',
        'Sim4Ansatz',
        'Sim9Ansatz',
        'Sim9CxAnsatz',
        'SpiderAnsatz',
        'StronglyEntanglingAnsatz',
        'Symbol',
        'lambdify',
        'TensorAnsatz',

        'CCGType',
        'CCGRule',
        'CCGRuleUseError',
        'CCGTree',

        'CCGParser',
        'BobcatParseError',
        'BobcatParser',
        'CCGBankParseError',
        'CCGBankParser',
        'DepCCGParseError',
        'DepCCGParser',
        'OncillaParseError',
        'OncillaParser',
        'WebParseError',
        'WebParser',

        'AtomicType',

        'VerbosityLevel',

        'Reader',
        'LinearReader',
        'TreeReader',
        'TreeReaderMode',
        'bag_of_words_reader',
        'cups_reader',
        'spiders_reader',
        'stairs_reader',
        'word_sequence_reader',

        'CollapseDomainRewriteRule',
        'CoordinationRewriteRule',
        'CurryRewriteRule',
        'DiagramRewriter',
        'RemoveCupsRewriter',
        'RemoveSwapsRewriter',
        'Rewriter',
        'RewriteRule',
        'SimpleRewriteRule',
        'UnifyCodomainRewriter',
        'UnknownWordsRewriteRule',

        'Tokeniser',
        'SpacyTokeniser',

        'Checkpoint',

        'Dataset',

        'Optimizer',
        'NelderMeadOptimizer',
        'RotosolveOptimizer',
        'SPSAOptimizer',

        'Model',
        'NumpyModel',
        'PennyLaneModel',
        'PytorchModel',
        'PytorchQuantumModel',
        'QuantumModel',
        'TketModel',

        'Trainer',
        'PytorchTrainer',
        'QuantumTrainer',

        'TnPathOptimizer',
        'CachedTnPathOptimizer',

        'BinaryCrossEntropyLoss',
        'CrossEntropyLoss',
        'LossFunction',
        'MSELoss',
]

from lambeq.backend import Symbol, lambdify
from lambeq import ansatz, core, rewrite, text2diagram, tokeniser, training
from lambeq.ansatz import (BaseAnsatz, CircuitAnsatz, IQPAnsatz, MPSAnsatz,
                           Sim14Ansatz, Sim15Ansatz, Sim4Ansatz, SpiderAnsatz,
                           StronglyEntanglingAnsatz, TensorAnsatz, Sim9Ansatz,
                           Sim9CxAnsatz)
from lambeq.core.globals import VerbosityLevel
from lambeq.core.types import AtomicType
from lambeq.rewrite import (CollapseDomainRewriteRule, CoordinationRewriteRule,
                            CurryRewriteRule, DiagramRewriter,
                            RemoveCupsRewriter, RemoveSwapsRewriter, Rewriter,
                            RewriteRule, SimpleRewriteRule,
                            UnifyCodomainRewriter, UnknownWordsRewriteRule)
from lambeq.text2diagram import (
        CCGType, CCGRule, CCGRuleUseError, CCGTree,
        CCGParser,
        BobcatParseError, BobcatParser,
        OncillaParseError, OncillaParser,
        CCGBankParseError, CCGBankParser,
        DepCCGParseError, DepCCGParser,
        WebParseError, WebParser,
        Reader, LinearReader, TreeReader, TreeReaderMode,
        bag_of_words_reader, cups_reader, spiders_reader,
        stairs_reader, word_sequence_reader)
from lambeq.tokeniser import Tokeniser, SpacyTokeniser
from lambeq.training import (Checkpoint, Dataset, Optimizer,
                             NelderMeadOptimizer, RotosolveOptimizer,
                             SPSAOptimizer, Model, NumpyModel,
                             PennyLaneModel, PytorchModel,
                             PytorchQuantumModel, QuantumModel,
                             TketModel, Trainer, PytorchTrainer,
                             CachedTnPathOptimizer, TnPathOptimizer,
                             QuantumTrainer, BinaryCrossEntropyLoss,
                             CrossEntropyLoss, LossFunction, MSELoss)
from lambeq.version import (version as __version__,
                            version_tuple as __version_info__)
