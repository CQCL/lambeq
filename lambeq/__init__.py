# Copyright 2021, 2022 Cambridge Quantum Computing Ltd.
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
        'text2diagram',
        'core',
        'pregroups',
        'reader',
        'rewrite',
        'tokeniser',
        'training',

        'BaseAnsatz',
        'CircuitAnsatz',
        'IQPAnsatz',
        'MPSAnsatz',
        'SpiderAnsatz',
        'Symbol',
        'TensorAnsatz',

        'CCGAtomicType',
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

        'Reader',
        'LinearReader',
        'TreeReader',
        'TreeReaderMode',
        'bag_of_words_reader',
        'cups_reader',
        'spiders_reader',
        'stairs_reader',
        'word_sequence_reader',

        'RewriteRule',
        'CoordinationRewriteRule',
        'CurryRewriteRule',
        'SimpleRewriteRule',
        'Rewriter',

        'Tokeniser',
        'SpacyTokeniser',

        'Checkpoint',

        'Dataset',

        'Optimizer',
        'SPSAOptimizer',

        'Model',
        'NumpyModel',
        'PytorchModel',
        'QuantumModel',
        'TketModel',

        'Trainer',
        'PytorchTrainer',
        'QuantumTrainer',
]

from lambeq.version import (version as __version__,
                            version_tuple as __version_info__)

from lambeq import (ansatz, bobcat, core, pregroups, rewrite,
                    text2diagram, tokeniser, training)
from lambeq.ansatz import (BaseAnsatz, CircuitAnsatz, IQPAnsatz, MPSAnsatz,
                           SpiderAnsatz, Symbol, TensorAnsatz)
from lambeq.core.types import AtomicType
from lambeq.core.globals import VerbosityLevel
from lambeq.pregroups import (diagram2str,
                              create_pregroup_diagram, is_pregroup_diagram,
                              remove_cups)
from lambeq.rewrite import (RewriteRule, CoordinationRewriteRule,
                            CurryRewriteRule, SimpleRewriteRule, Rewriter)
from lambeq.text2diagram import (
        CCGAtomicType, CCGRule, CCGRuleUseError, CCGTree,
        CCGParser,
        BobcatParseError, BobcatParser,
        CCGBankParseError, CCGBankParser,
        DepCCGParseError, DepCCGParser,
        WebParseError, WebParser,
        Reader, LinearReader, TreeReader, TreeReaderMode,
        bag_of_words_reader, cups_reader, spiders_reader,
        stairs_reader, word_sequence_reader)
from lambeq.tokeniser import Tokeniser, SpacyTokeniser
from lambeq.training import (Checkpoint, Dataset, Optimizer, SPSAOptimizer,
                             Model, NumpyModel, PytorchModel, QuantumModel,
                             TketModel, Trainer, PytorchTrainer,
                             QuantumTrainer)
