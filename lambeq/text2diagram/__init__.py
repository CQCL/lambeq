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

__all__ = ['CCGAtomicType',
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

           'LinearReader',
           'Reader',
           'TreeReader',
           'TreeReaderMode',
           'bag_of_words_reader',
           'cups_reader',
           'spiders_reader',
           'stairs_reader',
           'word_sequence_reader']

from lambeq.text2diagram.ccg_rule import CCGRule, CCGRuleUseError
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_types import CCGAtomicType

from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.bobcat_parser import BobcatParseError, BobcatParser
from lambeq.text2diagram.ccgbank_parser import CCGBankParseError, CCGBankParser
from lambeq.text2diagram.depccg_parser import DepCCGParseError, DepCCGParser
from lambeq.text2diagram.web_parser import WebParseError, WebParser

from lambeq.text2diagram.linear_reader import (LinearReader,
                                               cups_reader,
                                               stairs_reader,
                                               word_sequence_reader)
from lambeq.text2diagram.spiders_reader import (bag_of_words_reader,
                                                spiders_reader)
from lambeq.text2diagram.tree_reader import TreeReader, TreeReaderMode
