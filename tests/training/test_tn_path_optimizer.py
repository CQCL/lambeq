import pickle
import pytest
from unittest.mock import mock_open, patch

from lambeq import Checkpoint
from lambeq.backend.quantum import CX, H, Ket, Rx, Rz
from lambeq.training.tn_path_optimizer import (
    CachedTnPathOptimizer,
    ordered_nodes_contractor
)

diagram = Ket(0) @ Ket(0) >> H @ H >> CX >> Rx(0.5) @ Rz(0.2) >> CX
diagram2 = Ket(0) >> H >> Rx(0.5)


def test_path_finding():
    tn_path_optimizer = CachedTnPathOptimizer()
    assert len(tn_path_optimizer.cached_paths) == 0

    nodes, out_edges = diagram.to_tn()
    ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
    assert len(tn_path_optimizer.cached_paths) == 1, "Path is found and cached"

    nodes, out_edges = diagram.to_tn()
    ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
    assert len(tn_path_optimizer.cached_paths) == 1, "Previous path is reused"

    nodes, out_edges = diagram2.to_tn()
    ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
    assert len(tn_path_optimizer.cached_paths) == 2, "New path for new diagram"


@pytest.mark.parametrize("save_checkpoints", [True, False])
def test_tn_optimizer_checkpoint_loading(save_checkpoints):
    tn_path_optimizer = CachedTnPathOptimizer(save_checkpoints=save_checkpoints)
    assert isinstance(tn_path_optimizer, CachedTnPathOptimizer)

    nodes, out_edges = diagram.to_tn()
    ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
    assert len(tn_path_optimizer.cached_paths) == 1

    checkpoint = tn_path_optimizer.store_to_checkpoint(Checkpoint())

    path_optimizer_new = CachedTnPathOptimizer()
    path_optimizer_new.restore_from_checkpoint(checkpoint)
    assert isinstance(path_optimizer_new, CachedTnPathOptimizer)
    # If not saving paths to checkpoint, expect new paths to be empty
    if save_checkpoints:
        assert tn_path_optimizer.cached_paths == path_optimizer_new.cached_paths
        assert len(path_optimizer_new.cached_paths) == 1
    else:
        assert len(path_optimizer_new.cached_paths) == 0

def test_tn_optimizer_save_file():
    # Init empty file
    with patch('lambeq.training.tn_path_optimizer.open',
               mock_open(read_data=pickle.dumps({}))) as m:
        tn_path_optimizer = CachedTnPathOptimizer(save_file="fake/file.pkl")
        assert isinstance(tn_path_optimizer, CachedTnPathOptimizer)
        m.assert_called_with('fake/file.pkl', 'rb')

        # Add a contraction path
        nodes, out_edges = diagram.to_tn()
        ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
        assert len(tn_path_optimizer.cached_paths) == 1
        m.assert_called_with('fake/file.pkl', 'wb')


def test_tn_optimizer_load_save_file():
    # Generate some fake paths to save.
    tn_path_optimizer = CachedTnPathOptimizer()
    assert len(tn_path_optimizer.cached_paths) == 0

    nodes, out_edges = diagram.to_tn()
    ordered_nodes_contractor(nodes, tn_path_optimizer, out_edges)
    fake_paths = tn_path_optimizer.cached_paths

    with patch('lambeq.training.tn_path_optimizer.open',
               mock_open(read_data=pickle.dumps(fake_paths))) as m:
        path_optimizer_new = CachedTnPathOptimizer(save_file="fake/file.pkl")
        assert isinstance(path_optimizer_new, CachedTnPathOptimizer)
        m.assert_called_with('fake/file.pkl', 'rb')
        assert path_optimizer_new.cached_paths == fake_paths

        # Check that the new path gets correctly reused
        nodes, out_edges = diagram.to_tn()
        ordered_nodes_contractor(nodes, path_optimizer_new, out_edges)
        assert path_optimizer_new.cached_paths == fake_paths
