# Copyright 2021 Cambridge Quantum Computing Ltd.
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

__all__ = ['CCGParser']

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from discopy import Diagram

from lambeq.ccg2discocat.ccg_tree import CCGTree


class CCGParser(ABC):
    """Base class for CCG parsers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialise the CCG parser."""

    @abstractmethod
    def sentences2trees(
            self,
            sentences: Iterable[str],
            suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : iterable of str
            The sentences to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. (may contain :py:obj:`None` if exceptions
            are suppressed)

        """

    def sentence2tree(self,
                      sentence: str,
                      suppress_exceptions: bool = False) -> Optional[CCGTree]:
        """Parse a sentence into a :py:class:`.CCGTree`.

        Parameters
        ----------
        sentence : str
            The sentence to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.

        Returns
        -------
        CCGTree or None
            The parsed tree, or :py:obj:`None` on failure.

        """
        return self.sentences2trees([sentence],
                                    suppress_exceptions=suppress_exceptions)[0]

    def sentences2diagrams(
            self,
            sentences: Iterable[str],
            planar: bool = False,
            suppress_exceptions: bool = False) -> List[Optional[Diagram]]:
        """Parse multiple sentences into a list of discopy diagrams.

        Parameters
        ----------
        sentences : iterable of str
            The sentences to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        list of discopy.Diagram or None
            The parsed diagrams. (may contain :py:obj:`None` if
            exceptions are suppressed)

        """
        trees = self.sentences2trees(sentences,
                                     suppress_exceptions=suppress_exceptions)
        diagrams = []
        for tree in trees:
            if tree is not None:
                try:
                    diagrams.append(tree.to_diagram(planar=planar))
                except Exception as e:
                    if suppress_exceptions:
                        diagrams.append(None)
                    else:
                        raise e
            else:
                diagrams.append(None)
        return diagrams

    def sentence2diagram(
            self,
            sentence: str,
            planar: bool = False,
            suppress_exceptions: bool = False) -> Optional[Diagram]:
        """Parse a sentence into a DisCoPy diagram.

        Parameters
        ----------
        sentence : str
            The sentence to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if
            the sentence fails to parse, instead of raising an
            exception, returns :py:obj:`None`.

        Returns
        -------
        discopy.Diagram or None
            The parsed diagram, or :py:obj:`None` on failure.

        """
        return self.sentences2diagrams(
                [sentence],
                planar=planar,
                suppress_exceptions=suppress_exceptions)[0]
