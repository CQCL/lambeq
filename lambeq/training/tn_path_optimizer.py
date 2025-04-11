# Copyright 2021-2025 Cambridge Quantum Computing Ltd.
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
Opt-Einsum contractor for reusing previously computed paths.
"""

import functools
from pathlib import Path
import pickle
from typing import (
    Collection, Iterable, Sequence, TypeVar
)

import opt_einsum as oe     # type: ignore[import-untyped]
from tensornetwork import (
    AbstractNode, contract_between, contract_parallel,
    Edge, get_all_edges, get_subgraph_dangling
)
from tensornetwork.contractors.opt_einsum_paths import utils

from lambeq.training.checkpoint import Checkpoint


T = TypeVar('T')
ContractionPath = Collection[tuple[int, ...]]
ContractionKey = tuple[
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[tuple[str, int], ...]
]


def _dedup(seq: Iterable[T]) -> list[T]:
    """
    Remove duplicates from list while maintaining order of the items.
    Note: this method was taken from
    https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-
    from-a-list-while-preserving-order
    """
    seen: set[T] = set()
    # Alias add() so python doesn't have to check it each time we call it
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class TnPathOptimizer(oe.paths.PathOptimizer):
    """Opt-einsum custom optimizer."""
    _optimizer: functools.partial[ContractionPath]
    memory_limit: int | None

    def __init__(
        self,
        algorithm: str = 'auto',
        memory_limit: int | None = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        algorithm: :py:class:`str`, default: 'auto'
            Algorithm type to use when the path is not already cached.
            recommended options are:

            - ``auto`` (default): Fast; decent enough paths.
            - ``auto-hq``: Slow, but finds very high quality paths.
              Not recommended.
            - ``random-greedy``: Highly configurable, usually faster than
              auto-hq and finds better paths than auto.
        memory_limit: :py:class:`int`
            Limit the memory usage of the intermediate tensors. This is
            not recommended, as it will generally make the path finding
            much slower. If the size is a concern, use the random-greedy
            algorithm with ``minimize='size'`` instead.
        kwargs: Extra keyword arguments to pass to the fallback
            algorithm initializer. These will depend on the chosen
            fallback algorithm. For random-greedy, the following kwargs
            are available:

            - ``max_repeats``: int = 32
            - ``max_time``: float
            - ``minimize``: 'size' | 'flops' = 'flops'
            - ``parallel``: bool | int = False - whether to run trials in
              parallel. If a number is specified, use that many processes
              at once, otherwise use all available CPU cores.
        """
        self.memory_limit = memory_limit
        self._optimizer = functools.partial(
            oe.paths.get_path_fn(algorithm),
            **kwargs
        )

    def store_to_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        return checkpoint

    def restore_from_checkpoint(self, checkpoint: Checkpoint):
        """Load any information saved to a checkpoint.
        We do not expect to recover init kwargs from the checkpoint -
        the TnPathOptimizer should be initialised separately."""
        pass

    def __call__(
        self,
        inputs: list[set[Edge]],
        output: set[Edge],
        size_dict: dict[Edge, int],
        memory_limit: int | None = None,
        edge_list: list[Edge] | None = None
    ):
        return self._optimizer(
            inputs,
            output,
            size_dict,
            memory_limit=memory_limit or self.memory_limit
        )


class CachedTnPathOptimizer(TnPathOptimizer):
    """
    Opt-einsum custom optimizer.
    Cache computed paths for quick path lookup.
    """
    cached_paths: dict[ContractionKey, ContractionPath]
    filepath: Path | None
    save_checkpoints: bool

    checkpoint_cached_paths_key = 'tn_path_optimizer_cached_paths'

    def __init__(
        self,
        algorithm: str = 'auto-hq',
        save_file: Path | None = None,
        save_checkpoints: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        algorithm: :py:class:`str`, default: 'auto-hq'
            Fallback algorithm type to use when the path is not
            already cached. Recommended options are:

            - ``auto``: Fast, but paths are not that efficient.
              Use if you do not expect paths to be reused.
            - ``auto-hq`` (default): Slow, but finds very high
              quality paths.
            - ``random-greedy``: Highly configurable, usually
              faster than auto-hq and finds better paths than auto.
              Preferred option for larger networks where auto-hq
              is too slow.
        save_file: :py:class:`Path` or :py:class:`None`
            (Optional) filepath to save the cached paths to.
            File contents is updated with each new path.
        save_checkpoints: :py:class:`bool`, default: False
            Whether to include the cached paths in the checkpoints.
        kwargs: Extra keyword arguments to pass to the fallback
            algorithm initializer. These will depend on the chosen
            fallback algorithm. All fallback algorithm above accept a
            max_memory kwarg:

            - ``memory_limit``: (optional) int

            However, it is preferable to use random-greedy with
            ``minimize='size'`` for better performance.
            For random-greedy, the following kwargs are available:

            - ``max_repeats``: int = 32
            - ``max_time``: float
            - ``minimize``: 'size' | 'flops' = 'flops'
            - ``parallel``: bool | int = False - whether to run
              trials in parallel. If a number is specified, use
              that many processes at once, otherwise use all
              available CPU cores.
        """
        super().__init__(algorithm, **kwargs)
        self.cached_paths = {}
        self.filepath = save_file
        self.save_checkpoints = save_checkpoints

        if self.filepath is not None:
            try:
                with open(self.filepath, 'rb') as f:
                    self.cached_paths = pickle.load(f)
            except FileNotFoundError:
                # Make sure the path exists for future use.
                self.filepath.mkdir(parents=True, exist_ok=True)
            except EOFError:
                # No previous paths present; continue.
                pass

    def __call__(
        self,
        inputs: list[set[Edge]],
        output: set[Edge],
        size_dict: dict[Edge, int],
        memory_limit: int | None = None,
        edge_list: list[Edge] | None = None
    ) -> ContractionPath:
        if edge_list is None:
            raise ValueError(
                'Edge list must be supplied in order to cache paths.'
            )

        # Tensornetwork gives us edges with wacky names and arbitrary
        # order. Make them a single symbol instead, starting from 'a'
        # for the first input item.
        edge_map = {
            edge: oe.get_symbol(i)
            for i, edge in enumerate(edge_list)
        }
        # get a hashable key for this contraction
        sizes = tuple(
            (edge_map[edge], size_dict[edge])
            for edge in edge_list
        )
        input_tuples = tuple(
            tuple(sorted(edge_map[e] for e in s))
            for s in inputs
        )
        output_tuple = tuple(sorted(edge_map[e] for e in output))
        key = (input_tuples, output_tuple, sizes)

        if key not in self.cached_paths:
            # Couldn't find a saved path.
            # Use fallback and save it for next time.
            path = super().__call__(
                inputs, output, size_dict, memory_limit, edge_list
            )
            self.add_paths({key: path})

        return self.cached_paths[key]

    def add_paths(self, new_paths: dict[ContractionKey, ContractionPath]):
        """Append new paths to the current cache."""
        self.cached_paths = {
            **self.cached_paths,
            **new_paths
        }
        if self.filepath is not None:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.cached_paths, f)

    def store_to_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        checkpoint = super().store_to_checkpoint(checkpoint)
        if self.save_checkpoints:
            checkpoint.add_many({
                self.checkpoint_cached_paths_key: self.cached_paths,
            })
        return checkpoint

    def restore_from_checkpoint(self, checkpoint: Checkpoint):
        super().restore_from_checkpoint(checkpoint)
        if self.checkpoint_cached_paths_key in checkpoint:
            self.cached_paths = checkpoint[self.checkpoint_cached_paths_key]


def ordered_nodes_contractor(
    nodes: list[AbstractNode],
    algorithm: TnPathOptimizer,
    output_edge_order: Sequence[Edge] | None = None,
    ignore_edge_order: bool = False
) -> AbstractNode:
    """Copy tensornetwork's base contractor but preserve
    the node order for caching purposes.

    Parameters
    ----------
    nodes: :py:class:`list[AbstractNode]` A collection of connected nodes.
    algorithm: :py:class:`oe.paths.PathOptimizer | SavedTnOptimizer`
        `opt_einsum` contraction method to use.
    output_edge_order: :py:class:`Sequence[Edge] | None`
        An optional list of edges. Edges of the
        final node in `nodes_set`
        are reordered into `output_edge_order`;
        if final node has more than one edge,
        `output_edge_order` must be provided.
    ignore_edge_order: :py:class:`bool`
        An option to ignore the output edge
        order.

    Returns
    -------
    :py:class:`AbstractNode`
        Final node after full contraction.
    """
    nodes_set = set(nodes)
    edges = get_all_edges(nodes_set)
    # output edge order has to be determined before any contraction
    # (edges are refreshed after contractions)

    if not ignore_edge_order:
        if output_edge_order is None:
            output_edge_order = list(get_subgraph_dangling(nodes))
            if len(output_edge_order) > 1:
                raise ValueError(
                    'The final node after contraction has more than '
                    'one remaining edge. In this case `output_edge_order` '
                    'has to be provided.'
                )

        if set(output_edge_order) != get_subgraph_dangling(nodes):
            raise ValueError(
                'output edges are not equal to the remaining '
                'non-contracted edges of the final node.'
            )

    for edge in edges:
        if not edge.is_disabled:
            # if it is disabled we already contracted it
            if edge.is_trace():
                nodes_set.remove(edge.node1)
                nodes_set.add(contract_parallel(edge))

    if len(nodes_set) == 1:
        # There's nothing to contract.
        if ignore_edge_order:
            return list(nodes_set)[0]
        return list(nodes_set)[0].reorder_edges(output_edge_order)

    # Then apply `opt_einsum`'s algorithm
    path, nodes = _get_path_with_ordered_nodes(
        _dedup([n for n in nodes if n in nodes_set]),
        algorithm
    )
    for a, b in path:
        new_node = contract_between(
            nodes[a],
            nodes[b], allow_outer_product=True
        )
        nodes.append(new_node)
        nodes = utils.multi_remove(nodes, [a, b])

    # if the final node has more than one edge,
    # output_edge_order has to be specified
    final_node = nodes[0]  # nodes were connected, we checked this
    if not ignore_edge_order:
        final_node.reorder_edges(output_edge_order)
    return final_node


def _get_path_with_ordered_nodes(
    nodes: list[AbstractNode],
    algorithm: TnPathOptimizer
) -> tuple[Collection[tuple[int, ...]], list[AbstractNode]]:
    """Calculates the contraction paths using `opt_einsum`
    methods. A copy of the tensornetwork implementation,
    that uses a consistent node ordering.

    Parameters
    ----------
    nodes:
        An iterable of nodes.
    algorithm:
        `opt_einsum` method to use for calculating the contraction path.

    Returns
    -------
        The optimal contraction path as returned by `opt_einsum`.
    """
    input_sets = [set(node.edges) for node in nodes]
    output_set = get_subgraph_dangling(nodes)
    size_dict = {edge: edge.dimension for edge in get_all_edges(nodes)}
    # Fix an edge order, so we can find this same contraction again later
    edge_list = _dedup(
        e
        for n in nodes
        for e in n.edges
    )

    return algorithm(
            input_sets, output_set,
            size_dict, edge_list=edge_list
        ), nodes
