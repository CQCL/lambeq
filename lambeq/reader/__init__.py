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

__all__ = ['LinearReader', 'Reader', 'TreeReader', 'TreeReaderMode',
           'bag_of_words_reader', 'cups_reader', 'spiders_reader',
           'stairs_reader', 'word_sequence_reader']

from lambeq.reader.base import (Reader, LinearReader, bag_of_words_reader,
                                cups_reader, spiders_reader, stairs_reader,
                                word_sequence_reader)
from lambeq.reader.tree_reader import TreeReader, TreeReaderMode
