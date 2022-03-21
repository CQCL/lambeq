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
           'WebParser']

from lambeq.ccg2discocat.ccg_rule import CCGRule, CCGRuleUseError
from lambeq.ccg2discocat.ccg_tree import CCGTree
from lambeq.ccg2discocat.ccg_types import CCGAtomicType

from lambeq.ccg2discocat.ccg_parser import CCGParser
from lambeq.ccg2discocat.bobcat_parser import BobcatParseError, BobcatParser
from lambeq.ccg2discocat.ccgbank_parser import CCGBankParseError, CCGBankParser
from lambeq.ccg2discocat.depccg_parser import DepCCGParseError, DepCCGParser
from lambeq.ccg2discocat.web_parser import WebParseError, WebParser
