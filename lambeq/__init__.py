# Copyright 2021-2023 Cambridge Quantum Computing Ltd.
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
        'pregroups',
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
        'SpiderAnsatz',
        'StronglyEntanglingAnsatz',
        'Symbol',
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
        'WebParseError',
        'WebParser',

        'AtomicType',

        'VerbosityLevel',

        'diagram2str',
        'create_pregroup_diagram',
        'is_pregroup_diagram',
        'remove_cups',
        'remove_swaps',

        'Reader',
        'LinearReader',
        'TreeReader',
        'TreeReaderMode',
        'bag_of_words_reader',
        'cups_reader',
        'spiders_reader',
        'stairs_reader',
        'word_sequence_reader',

        'CoordinationRewriteRule',
        'CurryRewriteRule',
        'DiagramRewriter',
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
        'QuantumModel',
        'TketModel',

        'Trainer',
        'PytorchTrainer',
        'QuantumTrainer',

        'BinaryCrossEntropyLoss',
        'CrossEntropyLoss',
        'LossFunction',
        'MSELoss',
]

from lambeq import (ansatz, core, pregroups, rewrite,
                    text2diagram, tokeniser, training)
from lambeq.ansatz import (BaseAnsatz, CircuitAnsatz, IQPAnsatz, MPSAnsatz,
                           Sim14Ansatz, Sim15Ansatz, SpiderAnsatz,
                           StronglyEntanglingAnsatz, Symbol, TensorAnsatz)
from lambeq.core.globals import VerbosityLevel
from lambeq.core.types import AtomicType
from lambeq.pregroups import (create_pregroup_diagram, diagram2str,
                              is_pregroup_diagram, remove_cups, remove_swaps)
from lambeq.rewrite import (CoordinationRewriteRule, CurryRewriteRule,
                            DiagramRewriter, Rewriter, RewriteRule,
                            SimpleRewriteRule, UnifyCodomainRewriter,
                            UnknownWordsRewriteRule)
from lambeq.text2diagram import (
        CCGType, CCGRule, CCGRuleUseError, CCGTree,
        CCGParser,
        BobcatParseError, BobcatParser,
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
                             PennyLaneModel, PytorchModel, QuantumModel,
                             TketModel, Trainer, PytorchTrainer,
                             QuantumTrainer, BinaryCrossEntropyLoss,
                             CrossEntropyLoss, LossFunction, MSELoss)
from lambeq.version import (version as __version__,
                            version_tuple as __version_info__)
