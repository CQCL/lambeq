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

"""
Interface with discopy
======================
Module containing the functions to convert from and to discopy.
This work is based on DisCoPy (https://discopy.org/) which is released
under the BSD 3-Clause "New" or "Revised" License.

"""
from __future__ import annotations

from typing import cast, Type, TypeVar, Union

from packaging import version

from lambeq.backend import grammar as lg
from lambeq.backend import quantum as lq
from lambeq.backend import tensor as lt


MIN_DISCOPY_VERSION = '1.1.0'

try:
    import discopy
except ImportError as ie:
    raise ImportError(
        '`import discopy` failed. Please install discopy by '
        f'running `pip install "discopy>={MIN_DISCOPY_VERSION}"`.'
    ) from ie
else:
    if version.parse(discopy.__version__) < version.parse(MIN_DISCOPY_VERSION):
        raise DeprecationWarning(
            'Conversion from lambeq to discopy and vice versa '
            f'requires discopy>={MIN_DISCOPY_VERSION}. Please update discopy '
            f'by running `pip install "discopy>={MIN_DISCOPY_VERSION}"`.'
        )

from discopy import quantum as dq   # noqa: E402,I100
from discopy import tensor as dt    # noqa: E402
from discopy.grammar import pregroup as dg  # noqa: E402


_LAMBEQ_QUANTUM_BOX_TY = Union[type[lq.Box], lq.Box]
_DISCOPY_QUANTUM_BOX_TY = Union[type[dq.Box], dq.Box]
_QUANTUM_MAP_L2D_TY = dict[_LAMBEQ_QUANTUM_BOX_TY, _DISCOPY_QUANTUM_BOX_TY]
_QUANTUM_MAP_D2L_TY = dict[_DISCOPY_QUANTUM_BOX_TY, _LAMBEQ_QUANTUM_BOX_TY]
QUANTUM_MAPPINGS_L2D: _QUANTUM_MAP_L2D_TY = {lq.Discard: dq.Discard,
                                             lq.Encode: dq.Encode,
                                             lq.Measure: dq.Measure,
                                             lq.Bra: dq.Bra,
                                             lq.Ket: dq.Ket,
                                             lq.Sqrt: dq.gates.Sqrt,
                                             lq.Scalar: dq.gates.Scalar,
                                             lq.SWAP: dq.SWAP,
                                             lq.H: dq.H,
                                             lq.S: dq.S,
                                             lq.T: dq.T,
                                             lq.X: dq.X,
                                             lq.Y: dq.Y,
                                             lq.Z: dq.Z,
                                             lq.Rx: dq.Rx,
                                             lq.Ry: dq.Ry,
                                             lq.Rz: dq.Rz}

QUANTUM_MAPPINGS_D2L: _QUANTUM_MAP_D2L_TY = {
    val: key for key, val in QUANTUM_MAPPINGS_L2D.items()
}
_LAMBEQ_DIAGRAM_TY = Union[lg.Diagram, lq.Diagram, lt.Diagram]
_DISCOPY_DIAGRAM_TY = Union[dg.Diagram, dq.Circuit, dt.Diagram]
_DISCOPY_TY_VAR = TypeVar('_DISCOPY_TY_VAR', dg.Ty, dq.Ty, dt.Dim)
_LAMBEQ_TY_VAR = TypeVar('_LAMBEQ_TY_VAR', lg.Ty, lq.Ty, lt.Dim)
_DISCOPY_BOX_VAR = TypeVar('_DISCOPY_BOX_VAR', dg.Box, dq.Box, dt.Box)
_LAMBEQ_BOX_VAR = TypeVar('_LAMBEQ_BOX_VAR',
                          lg.Box, lq.Box, lt.Box, lq.Diagram)
_DISCOPY_ENTITY = TypeVar('_DISCOPY_ENTITY', dg.Ty, dq.Ty, dt.Dim, dg.Box,
                          dq.Box, dt.Box)


def _unwind_discopy_entity(entity: _DISCOPY_ENTITY) -> _DISCOPY_ENTITY:
    """Unwind a discopy entity."""
    if entity.z in (None, 0):
        return entity
    for _ in range(abs(entity.z)):
        entity = entity.l if entity.z > 0 else entity.r
    return entity


def _wind_discopy_entity(entity: _DISCOPY_ENTITY, z: int) -> _DISCOPY_ENTITY:
    """Wind a discopy entity."""
    for _ in range(abs(z)):
        entity = entity.r if z > 0 else entity.l
    return entity


def ty_l2d(ty: lg.Ty | lq.Ty | lt.Dim,
           target_type: Type[_DISCOPY_TY_VAR]) -> _DISCOPY_TY_VAR:

    converted = target_type()
    for subty in ty:

        z = subty.z
        subty = subty.unwind()

        if subty.category == lq.quantum:
            if subty == lq.bit:
                converted_type = dq.bit
            elif subty == lq.qubit:
                converted_type = dq.qubit
            else:
                converted_type = target_type(subty.name)
        elif subty.category == lg.grammar:
            converted_type = target_type(subty.name)
        elif subty.category == lt.tensor:
            converted_type = target_type(
                subty.dim[0])  # type: ignore[attr-defined]
        else:
            raise NotImplementedError(subty)

        converted_type = _wind_discopy_entity(converted_type, z)
        converted @= converted_type

    return converted


def ty_d2l(ty: dg.Ty | dq.Ty | dt.Dim,
           target_type: Type[_LAMBEQ_TY_VAR]) -> _LAMBEQ_TY_VAR:

    converted = target_type()
    for subty in ty:
        if not isinstance(subty, dt.Dim):
            z = subty.z
            subty = _unwind_discopy_entity(subty)
        else:
            z = 0

        if isinstance(subty, dq.Ty):
            if subty == dq.bit:
                converted @= lq.bit.rotate(z)
            elif subty == dq.qubit:
                converted @= lq.qubit.rotate(z)
            else:
                converted @= target_type(subty.name).rotate(z)
        elif isinstance(subty, dt.Dim):
            converted @= target_type(
                subty.inside[0]).rotate(z)
        elif isinstance(subty, dg.Ty):
            converted @= target_type(subty.name).rotate(z)
        else:
            raise NotImplementedError(subty)

    return converted


def convert_quantum_l2d(box: lq.Box) -> dq.Box:
    dq_box: _DISCOPY_QUANTUM_BOX_TY

    if isinstance(box, lq.Daggered):
        op = convert_quantum_l2d(box.dagger()).dagger()

    elif isinstance(box, lq.Controlled):
        op = dq.Controlled(controlled=convert_quantum_l2d(box.controlled),
                           distance=box.distance)

    elif isinstance(box, (lq.Rx, lq.Ry, lq.Rz, lq.Scalar, lq.Sqrt)):
        dq_box = cast(type[
            Union[dq.Rx, dq.Ry, dq.Rz, dq.gates.Scalar, dq.gates.Sqrt]
        ], QUANTUM_MAPPINGS_L2D[type(box)])
        op = dq_box(box.data)

    elif isinstance(box, (lq.Bra, lq.Ket)):
        dq_box = cast(type[Union[dq.Bra, dq.Ket]],
                      QUANTUM_MAPPINGS_L2D[type(box)])
        op = dq_box(box.bit)

    elif isinstance(box, (lq.Discard, lq.Encode, lq.Measure)):
        dq_box = cast(type[Union[dq.Discard, dq.Encode, dq.Measure]],
                      QUANTUM_MAPPINGS_L2D[type(box)])
        op = dq_box()

    else:
        try:
            op = cast(dq.Box, QUANTUM_MAPPINGS_L2D[box.unwind()])
            # Need to catch these `z` values because
            # `discopy.quantum.circuit.Box.rotate` rotates
            # even if passed arg is `None` or `0`.
            if box.z not in (None, 0):
                op = op.rotate(box.z)
        except KeyError:  # pragma: no cover
            raise NotImplementedError(box)

    return op


def convert_quantum_d2l(box: dq.Box) -> lq.Box | lq.Diagram:
    lq_box: _LAMBEQ_QUANTUM_BOX_TY

    if box.is_dagger:
        op = convert_quantum_d2l(box.dagger()).dagger()

    elif isinstance(box, dq.Controlled):
        controlled = convert_quantum_d2l(box.controlled)
        if isinstance(controlled, lq.Diagram):
            op = controlled
        else:
            op = lq.Controlled(controlled=controlled,
                               distance=box.distance)

    elif isinstance(box, (dq.Rx, dq.Ry, dq.Rz,
                          dq.gates.Scalar, dq.gates.Sqrt)):
        lq_box = cast(type[Union[lq.Rx, lq.Ry, lq.Rz, lq.Scalar, lq.Sqrt]],
                      QUANTUM_MAPPINGS_D2L[type(box)])
        op = lq_box(cast(float, box.data))

    elif isinstance(box, (dq.Bra, dq.Ket)):
        if box.bitstring:
            lq_box = cast(type[Union[lq.Bra, lq.Ket]],
                          QUANTUM_MAPPINGS_D2L[type(box)])

            op = lq_box(*box.bitstring)
        else:
            op = lq.Id()

    elif isinstance(box, (dq.Discard, dq.Encode)):
        lq_box = cast(type[Union[lq.Discard, lq.Encode]],
                      QUANTUM_MAPPINGS_D2L[type(box)])
        op = lq_box()

    elif isinstance(box, dq.Measure):
        if not box.destructive:
            raise NotImplementedError(f'Non-destructive measurement {box} '
                                      'not supported.')
        lq_box = cast(type[lq.Measure], QUANTUM_MAPPINGS_D2L[type(box)])
        op = lq_box()

    else:
        try:
            op = cast(lq.Box,
                      QUANTUM_MAPPINGS_D2L[_unwind_discopy_entity(box)])
            op = op.rotate(box.z)
        except KeyError:  # pragma: no cover
            raise NotImplementedError(box)

    return op


def convert_tensor_l2d(box: lt.Box) -> dt.Box:

    if isinstance(box, lt.Daggered):
        undaggered = cast(lt.Box, box.dagger())
        op = convert_tensor_l2d(undaggered).dagger()

    elif isinstance(box, (lt.Cap, lt.Cup)):
        cups_caps = {lt.Cap: dt.Cap, lt.Cup: dt.Cup}
        op = cups_caps[type(box)](
            left=ty_l2d(box.left, dt.Dim),
            right=ty_l2d(box.right, dt.Dim)
        )

    elif isinstance(box, lt.Swap):
        op = dt.Swap(left=ty_l2d(box.left, dt.Dim),
                     right=ty_l2d(box.right, dt.Dim))

    elif isinstance(box, lt.Spider):
        op = dt.Spider(n_legs_in=box.n_legs_in,
                       n_legs_out=box.n_legs_out,
                       typ=ty_l2d(box.type, dt.Dim))

    elif isinstance(box, lt.Box):
        op = dt.Box(name=box.name,
                    dom=ty_l2d(box.dom, dt.Dim),
                    cod=ty_l2d(box.cod, dt.Dim),
                    data=box.data,
                    z=box.z)

    else:  # pragma: no cover
        raise NotImplementedError(box)

    return op


def convert_tensor_d2l(box: dt.Box) -> lt.Box:

    if box.is_dagger:
        op = convert_tensor_d2l(box.dagger()).dagger()

    elif isinstance(box, dt.Cap):
        left = ty_d2l(box.left, lt.Dim)
        right = ty_d2l(box.right, lt.Dim)
        op = lt.Cap(left=left,
                    right=right,
                    is_reversed=left == right.l)

    elif isinstance(box, dt.Cup):
        left = ty_d2l(box.left, lt.Dim)
        right = ty_d2l(box.right, lt.Dim)
        op = lt.Cup(left=left,
                    right=right,
                    is_reversed=left == right.r)

    elif isinstance(box, dt.Swap):
        op = lt.Swap(left=ty_d2l(box.left, lt.Dim),
                     right=ty_d2l(box.right, lt.Dim))

    elif isinstance(box, dt.Spider):
        op = lt.Spider(type=ty_d2l(box.typ, lt.Dim),
                       n_legs_in=len(box.dom),
                       n_legs_out=len(box.cod))

    elif isinstance(box, dt.Box):
        op = lt.Box(name=box.name,
                    dom=ty_d2l(box.dom, lt.Dim),
                    cod=ty_d2l(box.cod, lt.Dim),
                    data=box.data,
                    z=box.z)

    else:  # pragma: no cover
        raise NotImplementedError(box)

    return op  # type: ignore[no-any-return]


def convert_grammar_l2d(box: lg.Box) -> dg.Box:

    if isinstance(box, lg.Daggered):
        op = convert_grammar_l2d(box.dagger()).dagger()

    elif isinstance(box, (lg.Cap, lg.Cup)):
        cups_caps = {lg.Cap: dg.Cap, lg.Cup: dg.Cup}
        op = cups_caps[type(box)](
            left=ty_l2d(box.left, dg.Ty),
            right=ty_l2d(box.right, dg.Ty)
        )

    elif isinstance(box, lg.Swap):
        op = dg.Swap(left=ty_l2d(box.left, dg.Ty),
                     right=ty_l2d(box.right, dg.Ty))

    elif isinstance(box, lg.Spider):
        op = dg.Spider(n_legs_in=box.n_legs_in,
                       n_legs_out=box.n_legs_out,
                       typ=ty_l2d(box.type, dg.Ty))

    elif isinstance(box, lg.Word):
        op = dg.Word(name=box.name,
                     cod=ty_l2d(box.cod, dg.Ty),
                     z=box.z)

    elif isinstance(box, lg.Box):
        op = dg.Box(name=box.name,
                    dom=ty_l2d(box.dom, dg.Ty),
                    cod=ty_l2d(box.cod, dg.Ty),
                    z=box.z)

    else:  # pragma: no cover
        raise NotImplementedError(box)

    return op


def convert_grammar_d2l(box: dg.Box) -> lg.Box:

    if box.is_dagger:
        op = convert_grammar_d2l(box.dagger()).dagger()

    elif isinstance(box, dg.Cap):
        left = ty_d2l(box.left, lg.Ty)
        right = ty_d2l(box.right, lg.Ty)
        op = lg.Cap(left=left,
                    right=right,
                    is_reversed=left == right.l)

    elif isinstance(box, dg.Cup):
        left = ty_d2l(box.left, lg.Ty)
        right = ty_d2l(box.right, lg.Ty)
        op = lg.Cup(left=left,
                    right=right,
                    is_reversed=left == right.r)

    elif isinstance(box, dg.Swap):
        op = lg.Swap(left=ty_d2l(box.left, lg.Ty),
                     right=ty_d2l(box.right, lg.Ty))

    elif isinstance(box, dg.Spider):
        op = lg.Spider(type=ty_d2l(box.typ, lg.Ty),
                       n_legs_in=len(box.dom),
                       n_legs_out=len(box.cod))

    elif isinstance(box, dg.Word):
        op = lg.Word(name=box.name,
                     cod=ty_d2l(box.cod, lg.Ty),
                     z=box.z)

    elif isinstance(box, dg.Box):
        op = lg.Box(name=box.name,
                    dom=ty_d2l(box.dom, lg.Ty),
                    cod=ty_d2l(box.cod, lg.Ty),
                    z=box.z)

    else:  # pragma: no cover
        raise NotImplementedError(box)

    return op


def box_l2d(box: lg.Box,
            target: type[_DISCOPY_BOX_VAR]) -> _DISCOPY_BOX_VAR:

    if target == dg.Box:
        assert isinstance(box, lg.Box)
        return convert_grammar_l2d(box)
    elif target == dq.Box:
        assert isinstance(box, lq.Box)
        return convert_quantum_l2d(box)
    elif target == dt.Box:
        assert isinstance(box, lt.Box)
        return convert_tensor_l2d(box)
    else:
        raise NotImplementedError(target)


def box_d2l(box: dg.Box,
            target: type[_LAMBEQ_BOX_VAR]) -> _LAMBEQ_BOX_VAR:

    if target == lg.Box:
        assert isinstance(box, dg.Box)
        return convert_grammar_d2l(box)  # type: ignore[return-value]
    elif target == lq.Box:
        assert isinstance(box, dq.Box)
        return convert_quantum_d2l(box)  # type: ignore[return-value]
    elif target == lt.Box:
        assert isinstance(box, dt.Box)
        return convert_tensor_d2l(box)  # type: ignore[return-value]
    else:
        raise NotImplementedError(target)


def to_discopy(diagram: _LAMBEQ_DIAGRAM_TY) -> _DISCOPY_DIAGRAM_TY:
    """Takes a :class:`lambeq.backend.grammar.Diagram`,
    :class:`lambeq.backend.quantum.Diagram`, or
    :class:`lambeq.backend.tensor.Diagram`, and converts it to a
    :class:`discopy.grammar.pregroup.Diagram`,
    :class:`discopy.quantum.Diagram`, or
    :class:`discopy.tensor.Diagram`, respectively.

    Parameters
    ----------
    diagram : `lambeq.backend.grammar.Diagram` |
        :class:`lambeq.backend.quantum.Diagram` |
        :class:`lambeq.backend.tensor.Diagram`
        The diagram to convert.

    Returns
    -------
    :class::class:`discopy.grammar.pregroup.Diagram` |
        :class:`discopy.quantum.Diagram` |
        :class:`discopy.tensor.Diagram`
        The converted diagram.

    """

    if isinstance(diagram, lq.Diagram):
        from discopy.quantum import Box, Ty, Id
    elif isinstance(diagram, lt.Diagram):
        from discopy.tensor import Box, Dim as Ty, Id
    elif isinstance(diagram, lg.Diagram):
        from discopy.grammar.pregroup import Box, Ty, Id
    else:
        raise NotImplementedError(diagram)

    box_factory = Box
    ty_factory = Ty
    id_factory = Id

    dcp_circ = id_factory(ty_l2d(diagram.dom, ty_factory))

    for layer in diagram.layers:
        left, box, right = layer.unpack()
        converted_left = ty_l2d(left, ty_factory)
        converted_right = ty_l2d(right, ty_factory)
        converted_box = box_l2d(box, box_factory)
        dcp_layer = converted_left @ converted_box @ converted_right
        dcp_circ >>= dcp_layer

    return dcp_circ


def from_discopy(diagram: _DISCOPY_DIAGRAM_TY) -> _LAMBEQ_DIAGRAM_TY:
    """Takes a :class:`discopy.grammar.pregroup.Diagram`,
    :class:`discopy.quantum.Diagram`, or
    :class:`discopy.tensor.Diagram`, and converts it to a
    :class:`lambeq.backend.grammar.Diagram`,
    :class:`lambeq.backend.quantum.Diagram`, or
    :class:`lambeq.backend.tensor.Diagram`, respectively.

    Parameters
    ----------
    diagram : :class:`discopy.grammar.pregroup.Diagram` |
        :class:`discopy.quantum.Diagram` |
        :class:`discopy.tensor.Diagram`
        The diagram to convert.

    Returns
    -------
    :class:`lambeq.backend.grammar.Diagram` |
        :class:`lambeq.backend.quantum.Diagram` |
        :class:`lambeq.backend.tensor.Diagram`
        The converted diagram.

    """

    if version.parse(discopy.__version__) < version.parse(MIN_DISCOPY_VERSION):
        raise DeprecationWarning(
            'Conversion from discopy to lambeq'
            f'requires discopy>={MIN_DISCOPY_VERSION}.'
        )

    if isinstance(diagram, dq.Circuit):
        from lambeq.backend.quantum import Box, Ty, Id
    elif isinstance(diagram, dt.Diagram):
        from lambeq.backend.tensor import (Box,  # type: ignore[assignment]
                                           Dim as Ty,
                                           Id)
    elif isinstance(diagram, dg.Diagram):
        from lambeq.backend.grammar import (Box,  # type: ignore[assignment]
                                            Ty,
                                            Id)
    else:
        raise NotImplementedError(diagram)

    box_factory = Box
    ty_factory = Ty
    id_factory = Id

    lam_circ = id_factory(ty_d2l(diagram.dom, ty_factory))

    for left, box, right in diagram:
        converted_left = ty_d2l(left, ty_factory)
        converted_right = ty_d2l(right, ty_factory)
        converted_box = box_d2l(box, box_factory)
        lam_layer = converted_left @ converted_box @ converted_right
        lam_circ >>= lam_layer

    return lam_circ
