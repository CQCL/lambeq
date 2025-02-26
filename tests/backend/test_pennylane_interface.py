from pytest import raises

import torch
import numpy as np

from lambeq.backend.quantum import *
from lambeq.backend.symbol import Symbol



def test_circuit_to_pennylane(capsys):
    bell_state = Diagram.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> Bra(0) @ bell_effect)[::-1]
    p_snake = snake.to_pennylane()
    p_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")

    assert np.allclose(p_snake.eval().numpy(), snake.eval())

    p_snake_prob = snake.to_pennylane(probabilities=True)
    snake_prob = (snake >> Measure())

    assert np.allclose(p_snake_prob.eval().numpy(), snake_prob.eval())

    no_open_snake = (bell_state @ Ket(0) >> Bra(0) @ bell_effect)[::-1]
    p_no_open_snake = no_open_snake.to_pennylane()
    p_no_open_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤0>\n")

    assert np.allclose(p_no_open_snake.eval().numpy(),
                       no_open_snake.eval())

    # probabilities should not be normalized if all wires are post-selected
    p_no_open_snake_prob = no_open_snake.to_pennylane(probabilities=True)

    assert np.allclose(p_no_open_snake_prob.eval().numpy(),
                       no_open_snake.eval())

    x, y, z = [Symbol(x) for x in 'xyz']
    symbols = [x, y, z]
    sym_symbols = [sym.unscaled for sym in symbols]
    weights = [torch.tensor(1.), torch.tensor(2.), torch.tensor(3.)]
    symbol_weight_map = dict(zip(sym_symbols, weights))

    var_circ = (Ket(0) >> Rx(0.552) >> Rz(x) >> Rx(0.917) >> Ket(0, 0, 0) @ qubit >>
                H @ qubit @ qubit @ qubit >> qubit @ H @ qubit @ qubit >>
                qubit @ qubit @ H @ qubit >> CRz(0.18) @ qubit @ qubit >>
                qubit @ CRz(y) @ qubit >> qubit @ qubit @ CX >>
                qubit @ qubit @ H @ qubit >> qubit @ qubit @ qubit @ Sqrt(2) @ qubit >>
                qubit @ qubit @ Bra(0, 0) >> Ket(0) @ qubit @ qubit >>
                Rx(0.446) @ qubit @ qubit >> Rz(0.256) @ qubit @ qubit >>
                Rx(z) @ qubit @ qubit >> CX @ qubit >> H @ qubit @ qubit >>
                qubit @ Sqrt(2) @ qubit @ qubit >> Bra(0, 0) @ qubit)

    p_var_circ = var_circ.to_pennylane()
    p_var_circ.initialise_concrete_params(symbol_weight_map)
    p_var_circ.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ──RX(2.80)──RZ(1.61)──RX(18.85)─╭●──H─┤0>\n"
         "1: ──H────────╭●───────────────────╰X────┤0>\n"
         "2: ──H────────╰RZ(1.13)─╭●───────────────┤  State\n"
         "3: ──H──────────────────╰RZ(12.57)─╭●──H─┤0>\n"
         "4: ──RX(3.47)──RZ(6.28)──RX(5.76)──╰X────┤0>\n")

    var_f = var_circ.lambdify(*symbols)
    conc_circ = var_f(*[a.item() for a in weights])

    assert np.allclose(p_var_circ.eval().numpy(),
                       conc_circ.eval())

    p_var_circ_prob = var_circ.to_pennylane(probabilities=True)
    p_var_circ_prob.initialise_concrete_params(symbol_weight_map)
    conc_circ_prob = (conc_circ >> Measure())

    assert (np.allclose(p_var_circ_prob.eval().numpy(),
                        conc_circ_prob.eval()))


def test_pennylane_circuit_mixed_error():
    bell_state = Diagram.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> Bra(0) @ bell_effect)[::-1]
    snake = (snake >> Measure())
    with raises(ValueError):
        snake.to_pennylane()


def test_pennylane_circuit_mixed_warning(capsys):
    bell_state = Diagram.caps(qubit, qubit)
    bell_discarded = bell_state >> Discard() @ Id(qubit)
    _ = bell_discarded.to_pennylane()
    captured = capsys.readouterr()
    assert captured.err == ('Warning: Circuit includes both discards and open '
                            'codomain wires. All open wires will be discarded '
                            'during conversion\n')


def test_pennylane_circuit_draw(capsys):
    bell_state = Diagram.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> Bra(0) @ bell_effect)[::-1]
    p_circ = snake.to_pennylane()
    p_circ.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")


def test_pennylane_ops():
    ops = [X, Y, Z, S, T, H, CX, CZ]

    for op in ops:
        lamb = (Id().tensor(*([Ket(0)] * len(op.dom))) >> op).eval()
        plane = op.to_pennylane().eval().numpy()

        assert np.allclose(lamb, plane)


def test_pennylane_parameterized_ops():
    ops = [Rx, Ry, Rz, CRx, CRz]

    for op in ops:
        p_op = op(0.5)
        lamb = (Id().tensor(*([Ket(0)] * len(p_op.dom))) >> p_op).eval()
        plane = p_op.to_pennylane().eval().numpy()

        assert np.allclose(lamb, plane, atol=10e-5)


def test_pennylane_devices():
    bell_state = Diagram.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ qubit >> Bra(0) @ bell_effect)[::-1]

    # Honeywell backend only compatible when `probabilities=True`
    h_backend = {'backend': 'honeywell.hqs', 'device': 'H1-1E'}
    h_circ = snake.to_pennylane(probabilities=True, backend_config=h_backend)
    assert h_circ._device is not None
    with raises(ValueError):
        h_circ = snake.to_pennylane(backend_config=h_backend)

    # Device must be specified when using Honeywell backend
    h_backend_corrupt = {'backend': 'honeywell.hqs'}
    with raises(ValueError):
        h_circ = snake.to_pennylane(probabilities=True,
                                    backend_config=h_backend_corrupt)

    aer_backend = {'backend': 'qiskit.aer',
                   'device': 'aer_simulator_statevector'}
    aer_circ = snake.to_pennylane(backend_config=aer_backend)
    assert aer_circ._device is not None

    # `aer_simulator` is not compatible with state outputs
    aer_backend_corrupt = {'backend': 'qiskit.aer', 'device': 'aer_simulator'}
    with raises(ValueError):
        aer_circ = snake.to_pennylane(backend_config=aer_backend_corrupt)


def test_pennylane_uninitialized():
    x, y, z = [Symbol(x) for x in 'xyz']
    var_circ = (Ket(0) >> Rx(0.552) >> Rz(x) >> Rx(0.917) >> Ket(0, 0, 0) @ qubit >>
                H @ qubit @ qubit @ qubit >> qubit @ H @ qubit @ qubit >>
                qubit @ qubit @ H @ qubit >> CRz(0.18) @ qubit @ qubit >>
                qubit @ CRz(y) @ qubit >> qubit @ qubit @ CX >>
                qubit @ qubit @ H @ qubit >> qubit @ qubit @ qubit @ Sqrt(2) @ qubit >>
                qubit @ qubit @ Bra(0, 0) >> Ket(0) @ qubit @ qubit >>
                Rx(0.446) @ qubit @ qubit >> Rz(0.256) @ qubit @ qubit >>
                Rx(z) @ qubit @ qubit >> CX @ qubit >> H @ qubit @ qubit >>
                qubit @ Sqrt(2) @ qubit @ qubit >> Bra(0, 0) @ qubit)
    p_var_circ = var_circ.to_pennylane()

    with raises(ValueError):
        p_var_circ.draw()

    with raises(ValueError):
        p_var_circ.eval()


def test_pennylane_parameter_reference():
    x = Symbol('x')
    p = torch.nn.Parameter(torch.tensor(1.))
    symbol_weight_map = {x.unscaled: p}

    circ = Rx(x)
    p_circ = circ.to_pennylane()
    p_circ.initialise_concrete_params(symbol_weight_map)

    assert p is p_circ._concrete_params[0][0]

    with torch.no_grad():
        p.add_(1.)

    assert p_circ._concrete_params[0][0] == p

    with torch.no_grad():
        p.add_(-2.)

    assert p_circ._concrete_params[0][0] == p


def test_pennylane_gradient_methods():
    x, y, z = [Symbol(x) for x in 'xyz']
    symbols = [x, y, z]
    sympy_symbols = [sym.unscaled for sym in symbols]

    var_circ = (Ket(0) >> Rx(0.552) >> Rz(x) >> Rx(0.917) >> Ket(0, 0, 0) @ qubit >>
                H @ qubit @ qubit @ qubit >> qubit @ H @ qubit @ qubit >>
                qubit @ qubit @ H @ qubit >> CRz(0.18) @ qubit @ qubit >>
                qubit @ CRz(y) @ qubit >> qubit @ qubit @ CX >>
                qubit @ qubit @ H @ qubit >> qubit @ qubit @ qubit @ Sqrt(2) @ qubit >>
                qubit @ qubit @ Bra(0, 0) >> Ket(0) @ qubit @ qubit >>
                Rx(0.446) @ qubit @ qubit >> Rz(0.256) @ qubit @ qubit >>
                Rx(z) @ qubit @ qubit >> CX @ qubit >> H @ qubit @ qubit >>
                qubit @ Sqrt(2) @ qubit @ qubit >> Bra(0, 0) @ qubit)

    for diff_method in ['backprop', 'parameter-shift', 'finite-diff']:

        weights = [torch.tensor(1., requires_grad=True),
                   torch.tensor(2., requires_grad=True),
                   torch.tensor(3., requires_grad=True)]
        symbol_weight_map = dict(zip(sympy_symbols, weights))

        p_var_circ = var_circ.to_pennylane(probabilities=True,
                                           diff_method=diff_method)
        p_var_circ.initialise_concrete_params(symbol_weight_map)

        loss = p_var_circ.eval().norm(dim=0, p=2)
        loss.backward()
        assert weights[0].grad is not None

    for diff_method in ['backprop']:

        weights = [torch.tensor(1., requires_grad=True),
                   torch.tensor(2., requires_grad=True),
                   torch.tensor(3., requires_grad=True)]
        symbol_weight_map = dict(zip(sympy_symbols, weights))

        p_var_circ = var_circ.to_pennylane(probabilities=False,
                                           diff_method=diff_method)
        p_var_circ.initialise_concrete_params(symbol_weight_map)

        loss = p_var_circ.eval().norm(dim=0, p=2)
        loss.backward()
        assert weights[0].grad is not None
