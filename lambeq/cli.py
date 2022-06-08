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

"""
Command-line Interface
=============
Command-line interface for the lambeq toolkit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from collections.abc import Iterator, Sequence
import inspect
import json
import os
from pathlib import Path
import pickle
from typing import Any, Optional, Union
import yaml

import lambeq
from lambeq.ansatz import BaseAnsatz
from lambeq.ansatz.circuit import IQPAnsatz, CircuitAnsatz
from lambeq.ansatz.tensor import TensorAnsatz, SpiderAnsatz, MPSAnsatz
from lambeq.pregroups import text_printer
from lambeq.pregroups.utils import is_pregroup_diagram
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.bobcat_parser import BobcatParser
from lambeq.text2diagram.depccg_parser import DepCCGParser
from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.linear_reader import (cups_reader,
                                               stairs_reader)
from lambeq.text2diagram.spiders_reader import spiders_reader
from lambeq.text2diagram.tree_reader import TreeReader
from lambeq.tokeniser import SpacyTokeniser

import discopy

AVAILABLE_PARSERS: dict[str, type[CCGParser]] = {'bobcat': BobcatParser,
                                                 'depccg': DepCCGParser}

AVAILABLE_READERS: dict[str, Union[Reader, type[Reader]]] = {
        'spiders': spiders_reader,
        'stairs': stairs_reader,
        'cups': cups_reader,
        'tree': TreeReader}

AVAILABLE_ANSATZE: dict[str, type[BaseAnsatz]] = {'iqp': IQPAnsatz,
                                                  'tensor': TensorAnsatz,
                                                  'spider': SpiderAnsatz,
                                                  'mps': MPSAnsatz}

AVAILABLE_IMAGE_TYPES: list[str] = ['png', 'pdf', 'jpeg', 'jpg', 'eps',
                                    'pgf', 'ps', 'raw', 'rgba', 'svg',
                                    'svgz', 'tif', 'tiff']

DEFAULT_ARG_VALUES: dict[str, str] = {'output_format': 'text-unicode',
                                      'image_format': 'png', 
                                      'mode': 'string-diagram'}


class ArgumentList:
    """Class for passing arguments of type key=value to argparse.
    Constructed from a list of tuples (argument name, type, default value),
    where default value has to be of the specified type or None."""

    def __init__(self, choices: list[tuple[str, type, Any]]) -> None:
        self.valid_args = {k: v for k, v, _ in choices}
        self.default_values = {k: v for k, _, v in choices}
        for k, v, default in choices:
            if default is not None and not isinstance(default, v):
                raise ValueError(f'Default value of {k}, {default} is not '
                                 f'an instance of {v.__name__}.')

    def __contains__(self, argument: str) -> bool:
        key, value = argument.split('=')
        try:
            _ = self.valid_args[key](value)
            return True
        except (KeyError, ValueError):
            return False

    def __call__(self, argument: str) -> tuple[str, Any]:
        key, value = argument.split('=')
        return key, self.valid_args[key](value)

    def __iter__(self) -> Iterator[str]:
        for k, v in self.valid_args.items():
            yield f'{k}=<{v.__name__}> (default: {self.default_values[k]})'

    def all_options(self) -> str:
        options = []
        for k, v in self.valid_args.items():
            options.append(f'{k}=<{v.__name__}> (default: '
                           f'{self.default_values[k]})')
        return ', '.join(options)


class ArgumentListValidator(argparse.Action):
    """Class that checks the validity of ArgumentList inputs.
    Made compatible with argparse."""
    def __init__(self,
                 option_strings: list[str],
                 dest: Any,
                 choices: ArgumentList,
                 nargs: str = '*',
                 **kwargs: Any) -> None:
        self.available_args = choices
        super().__init__(option_strings,
                         dest,
                         nargs=nargs,
                         choices=choices,
                         **kwargs)

    def __call__(self,
                 parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: Optional[str] = None) -> None:
        if values is None:  # pragma: no cover
            values = []
        elif isinstance(values, str):  # pragma: no cover
            values = values.split()
        parsed_args = {}
        for argument in values:
            key, value = self.available_args(argument)
            parsed_args[key] = value
        for key, value in self.available_args.default_values.items():
            if key not in parsed_args and value is not None:
                parsed_args[key] = value
        setattr(namespace, self.dest, parsed_args)


def prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
             description='Command-line interface for lambeq.',
             prog='lambeq')

    parser.add_argument('input_sentence',
                        nargs='?',
                        default='',
                        help='Sentence to parse.')
    parser.add_argument(
            '-m',
            '--mode',
            type=str,
            choices=['string-diagram', 'ccg'],
            default=DEFAULT_ARG_VALUES['mode'],
            help='Mode used for the output. Default value: '
                 f'{DEFAULT_ARG_VALUES["mode"]}')
    parser.add_argument('-i', '--input_file', type=str, help='File to parse.')

    output_group = parser.add_argument_group(
                          title='Output',
                          description='Options related to output format.')
    output_group.add_argument(
            '-f',
            '--output_format',
            type=str,
            choices=['json', 'pickle', 'text-unicode', 'text-ascii', 'image'],
            help='Format of the output. Use `json` and `pickle` to store the '
                 'lambeq / DisCoPy objects in the respective formats, or '
                 '`text-unicode`, `text-ascii` and `image` to store directly '
                 'the derivations in diagrammatic form. Default value: '
                 f'{DEFAULT_ARG_VALUES["output_format"]}')
    output_group.add_argument(
            '-g',
            '--image_format',
            type=str,
            choices=AVAILABLE_IMAGE_TYPES,
            help='When `image` is selected as `output_format`, this option '
                 'specifies the required image type. It does not have any '
                 'effect when any other option is selected as '
                 '`output_format`. Default value: '
                 f'{DEFAULT_ARG_VALUES["image_format"]}')
    output_arglist = ArgumentList([('fig_width', int, None),
                                   ('fig_height', int, None),
                                   ('fontsize', int, None)])

    output_group.add_argument(
            '-u',
            '--output_options',
            nargs='*',
            default={},
            choices=output_arglist,
            metavar='KEY=VAR',
            action=ArgumentListValidator,
            help='A list of `keyword=value` items that define options for the '
                 'output format. Available options are '
                 f'{output_arglist.all_options()}.')

    store_group = output_group.add_mutually_exclusive_group(required=False)
    store_group.add_argument(
            '-o',
            '--output_file',
            type=str,
            help='File to write the output. When `output_format` is `json`, '
                 '`text-ascii`, or `text-unicode` and this argument is '
                 'not provided, lambeq will output to `stdout`. '
                 'This argument is ignored when `output_format` is `image`, '
                 'in which case `output_dir` needs to be provided.')
    store_group.add_argument(
            '-d',
            '--output_dir',
            type=str,
            help='When `image` is selected as `output_format`, this option '
                 'specifies the directory where the image files would be '
                 'stored. It does not have effect when any other option is '
                 'selected as `output_format`.')

    parser_group = parser.add_argument_group(
                          title='Parser',
                          description='Options related to parser.')
    parser_group.add_argument(
            '-p',
            '--parser',
            type=str,
            default=None,
            choices=AVAILABLE_PARSERS.keys(),
            help='Choice of a parser. Mutually exclussive with using a '
                 'reader. If `None`, BobcatParser is used.')
    parser_group.add_argument(
            '-t',
            '--tokenise',
            default=False,
            action='store_true',
            help='Tokenises the input before sending to parser. If not used, '
                 'the parser assumes that text is already tokenised.')
    parser_group.add_argument(
            '-s',
            '--split_sentences',
            default=False,
            action='store_true',
            help='Use SpaCy sentence splitting to split the text into '
                 'sentences. Not required if only one sentence is provided or '
                 'if sentences are already given one per line.')
    parser_group.add_argument(
            '-r',
            '--reader',
            type=str,
            default=None,
            choices=AVAILABLE_READERS.keys(),
            help='Choice of a reader. Mutually exclusive with using a parser.')
    parser_group.add_argument(
            '-c',
            '--root_categories',
            nargs='*',
            metavar='ROOT_CAT',
            help='A list of acceptable categories at the root of the diagram.')

    rewrite_group = parser.add_argument_group(
                           title='Rewriter',
                           description='Rewrite options.')
    rewriter_options = lambeq.rewrite.Rewriter.available_rules()
    rewrite_group.add_argument(
            '-w',
            '--rewrite_rules',
            nargs='*',
            choices=rewriter_options,
            metavar='REWRITE_RULE',
            help='A list of rewrite rules. Available options: '
                 f'{rewriter_options}')

    ansatz_group = parser.add_argument_group(
                          title='Ansätze',
                          description='Options related to ansatz choices.')
    ansatz_group.add_argument(
            '-a',
            '--ansatz',
            type=str,
            default=None,
            choices=AVAILABLE_ANSATZE.keys(),
            help='Ansatz to be used. This determines if the result will be a '
                 'quantum circuit or a tensor network.')
    ansatz_arglist = ArgumentList([('dim_n', int, 2),
                                   ('dim_s', int, 2),
                                   ('n_layers', int, 2),
                                   ('n_single_qubit_params', int, 3),
                                   ('bond_dim', int, 3),
                                   ('max_order', int, 3)])

    ansatz_group.add_argument(
            '-n',
            '--ansatz_options',
            nargs='*',
            choices=ansatz_arglist,
            action=ArgumentListValidator,
            default={},
            metavar='KEY=VAR',
            help='A list of `keyword=value` items that define options for the '
                 'selected ansatz. Available options are '
                 f'{ansatz_arglist.all_options()}.')

    config_group = parser.add_argument_group(
                          title='Configuration',
                          description='Options for storing and loading the '
                                      'command-line arguments to/from files.')
    config_group.add_argument(
            '-y',
            '--store_args',
            type=str,
            help='File to store the parameters in YAML format for future use.')
    config_group.add_argument(
            '-l',
            '--load_args',
            type=str,
            help='Load and use a set of stored parameters from the specified '
                 'file.')
    return parser


def validate_args(cl_args: argparse.Namespace) -> None:
    """Checks if arguments are consistent and throws an error if not."""
    if cl_args.input_sentence != '' and cl_args.input_file is not None:
        raise ValueError('Both input file and input text were provided. '
                         'Only provide one at the time.')
    if (cl_args.output_dir is None and cl_args.output_file is None
            and cl_args.output_format not in ['text-ascii', 'text-unicode']):
        raise ValueError(f'{cl_args.output_format} output selected, but '
                         'no output file specified. '
                         'Use --output_file or --output_dir argument.')
    if cl_args.parser is not None and cl_args.reader is not None:
        raise ValueError('Reader and parser cannot be used at the same time.')
    if len(cl_args.output_options) > 0 and cl_args.output_format != 'image':
        raise ValueError('You have listed --output_options that are not '
                         f'used by the output type {cl_args.output_format}.')
    if cl_args.output_dir is not None and cl_args.output_format != 'image':
        raise ValueError(f'{cl_args.output_format} output type incompatible '
                         'with --output_dir option. Did you mean '
                         '--output_file?')
    if cl_args.mode == 'ccg':
        if any(v is not None for v in [cl_args.ansatz, cl_args.reader, 
                                       cl_args.rewrite_rules]):
            raise ValueError('Readers, rewrite rules, or ansatze cannot be '
                             'applied to CCG diagrams. In order to use them, '
                             'be sure `mode` is set to `string-diagram`.')
        if cl_args.output_format == 'image':
            raise ValueError('Generating binary images from CCG diagrams is '
                             'currently not supported. Try text forms, `json` '
                             'or `pickle`.')
    if cl_args.output_format in ['text-ascii', 'text-unicode', 'json']:
        if cl_args.mode == 'string-diagram':
            if cl_args.ansatz is not None:
                raise ValueError('Only pregroup diagrams can be stored in '
                                f'{cl_args.output_format} format. Use pickle or '
                                'image format or remove the --ansatz argument.')
            if cl_args.rewrite_rules is not None:
                raise ValueError('Only pregroup diagrams can be stored in '
                                f'{cl_args.output_format} format. Use pickle or '
                                'image format or remove the --rewrite_rules '
                                'argument.')
            if cl_args.reader in ['spiders', 'stairs', 'tree']:
                raise ValueError('Only pregroup diagrams can be stored in '
                                f'{cl_args.output_format} format. '
                                f'{cl_args.reader} reader does not return '
                                'pregroup diagrams. '
                                'Use pickle or image format or use a different '
                                'reader/parser.')


class CLIModule(ABC):
    """Base class for all modules in the CLI pipeline."""

    @abstractmethod
    def __call__(self, cl_args: argparse.Namespace, module_input: Any) -> Any:
        """Runs the relevant part of the pipeline and
        returns the input to the next module in the pipeline."""


class FileReaderModule(CLIModule):
    """Reads the text from the input file or the command line."""
    def __call__(self,
                 cl_args: argparse.Namespace,
                 module_input: Any = None) -> str:
        if cl_args.input_sentence != '':
            return cl_args.input_sentence  # type: ignore[no-any-return]
        elif cl_args.input_file is not None:
            with open(cl_args.input_file, 'r') as f:
                return f.read().strip()
        else:
            return input('Text: ').strip()


class ParserModule(CLIModule):
    """Tokenizes the text and parses it using the parser/reader of choice.
    Returns a list of pregroup diagrams."""

    def __call__(self,
                 cl_args: argparse.Namespace,
                 module_input: str) -> Union[list[discopy.Diagram], 
                                             list[CCGTree]]:
        if cl_args.split_sentences or cl_args.tokenise:
            tokeniser = SpacyTokeniser()
        sentences: Union[list[str], list[list[str]]]
        if cl_args.split_sentences:
            sentences = tokeniser.split_sentences(module_input)
        else:
            sentences = module_input.split('\n')
        if cl_args.tokenise:
            sentences = tokeniser.tokenise_sentences(sentences)
        if cl_args.reader is not None:
            reader = AVAILABLE_READERS[cl_args.reader.casefold()]
            if not isinstance(reader, Reader):
                reader = reader()
            return reader.sentences2diagrams(sentences,
                                             tokenised=cl_args.tokenise)
        elif cl_args.parser is not None:
            parser = AVAILABLE_PARSERS[cl_args.parser](
                    root_cats=cl_args.root_categories)
        else:
            parser = AVAILABLE_PARSERS['bobcat'](
                    root_cats=cl_args.root_categories)

        if cl_args.mode == 'ccg':
            trees = parser.sentences2trees(sentences, 
                                           tokenised=cl_args.tokenise)
            return [t.without_trivial_unary_rules() 
                    if t else None for t in trees]
        else:
            return parser.sentences2diagrams(sentences, 
                                             tokenised=cl_args.tokenise)


class RewriterModule(CLIModule):
    """Applies any rewriting rules to the diagram and normalizes it."""
    def __call__(
            self,
            cl_args: argparse.Namespace,
            module_input: list[discopy.Diagram]) -> list[discopy.Diagram]:
        if cl_args.rewrite_rules is None:
            return module_input
        rewriter = lambeq.rewrite.Rewriter(cl_args.rewrite_rules)
        return [rewriter(diagram).normal_form() for diagram in module_input]


class AnsatzModule(CLIModule):
    """Applies an ansatz to the diagram and returns a quantum circuit
       or a tensor diagram."""
    def __call__(self,
                 cl_args: argparse.Namespace,
                 module_input: list[discopy.Diagram]) -> list[discopy.Diagram]:
        N = lambeq.core.types.AtomicType.NOUN
        S = lambeq.core.types.AtomicType.SENTENCE
        if cl_args.ansatz is None:
            return module_input
        n_dim = cl_args.ansatz_options['dim_n']
        s_dim = cl_args.ansatz_options['dim_s']

        ansatz_type: type[BaseAnsatz]
        ansatz_type = AVAILABLE_ANSATZE[cl_args.ansatz.casefold()]

        supported_args, *_ = inspect.getfullargspec(ansatz_type.__init__)
        remaining_args = {k: v for k, v in cl_args.ansatz_options.items()
                          if k in supported_args}

        ansatz: BaseAnsatz
        if issubclass(ansatz_type, CircuitAnsatz):
            ansatz = ansatz_type({N: n_dim,
                                  S: s_dim},
                                 **remaining_args)
        elif issubclass(ansatz_type, TensorAnsatz):
            ansatz = ansatz_type({N: discopy.Dim(n_dim),
                                  S: discopy.Dim(s_dim)},
                                 **remaining_args)
        return [ansatz(diagram) for diagram in module_input]


class CCGTreeSaveModule(CLIModule):
    """Outputs CCG trees to text, json, and pickle formats."""
    def __call__(self,
                 cl_args: argparse.Namespace,
                 module_input: list[CCGTree]) -> None:
        if cl_args.output_format == 'json':
            with open(cl_args.output_file, 'w') as f:
                json.dump([t.to_json() for t in module_input], f)
        elif cl_args.output_format in ['text-ascii', 'text-unicode']:
            ascii_art = [t.deriv(use_ascii=(
                                    cl_args.output_format == 'text-ascii')) 
                            for t in module_input]
            if cl_args.output_file is not None:
                with open(cl_args.output_file, 'w') as f:
                    f.write('\n'.join(ascii_art))
            else:
                print('\n'.join(ascii_art))
        elif cl_args.output_format == 'pickle':
            if cl_args.output_file is not None:
                with open(cl_args.output_file, 'wb') as h:
                    pickle.dump(module_input, h)    


class DiagramSaveModule(CLIModule):
    """Outputs strings diagrams to text, json, pickle, and image formats.""" 
    def __call__(self,
                 cl_args: argparse.Namespace,
                 module_input: list[discopy.Diagram]) -> None:
        if cl_args.output_format in ['text-ascii', 'text-unicode']:
            for i in range(len(module_input)):
                if not is_pregroup_diagram(module_input[i]):
                    module_input[i] = discopy.grammar.normal_form(
                                        module_input[i])
                    if not is_pregroup_diagram(module_input[i]):
                        raise ValueError(  # pragma: no cover
                                'Output format is set to '
                                f'{cl_args.output_fomat} but '
                                f'parsing sentence no. {i+1} did not '
                                'produce a valid pregroup diagram.')
        if cl_args.output_format == 'json':
            with open(cl_args.output_file, 'w') as f:
                json.dump([d.to_tree() for d in module_input], f)
        elif cl_args.output_format in ['text-ascii', 'text-unicode']:
            printer = text_printer.TextDiagramPrinter(use_ascii=(
                                    cl_args.output_format == 'text-ascii'))
            ascii_art = [printer.diagram2str(diag) for diag in module_input]
            if cl_args.output_file is not None:
                with open(cl_args.output_file, 'w') as f:
                    f.write('\n'.join(ascii_art))
            else:
                print('\n'.join(ascii_art))
        elif cl_args.output_format == 'pickle':
            if cl_args.output_file is not None:
                with open(cl_args.output_file, 'wb') as h:
                    pickle.dump(module_input, h)
        elif cl_args.output_format == 'image':
            if cl_args.output_dir is not None:
                Path(cl_args.output_dir).mkdir(parents=True, exist_ok=True)
            elif len(module_input) > 1:
                raise ValueError('Multiple sentences detected in the input.'
                                 'It is not possible to save multiple images '
                                 'to a single file. To save multiple images, '
                                 'use --output_dir.')
            for i, diagram in enumerate(module_input):
                if cl_args.output_dir is not None:
                    img_id = str(i+1).zfill(len(str(len(module_input))))
                    image_path = os.path.join(
                                   cl_args.output_dir,
                                   f'diagram_{img_id}.{cl_args.image_format}')
                else:
                    image_path = cl_args.output_file
                    file_name, file_extension = os.path.splitext(image_path)
                    if file_extension not in AVAILABLE_IMAGE_TYPES:
                        image_path = f'{file_name}.{cl_args.image_format}'
                draw_args: dict[str, Any] = {'path': image_path}
                if 'fig_width' in cl_args.output_options and\
                   'fig_height' in cl_args.output_options and\
                   len(module_input) == 1:
                    draw_args['figsize'] = (
                            cl_args.output_options['fig_width'],
                            cl_args.output_options['fig_height'])
                if 'fontsize' in cl_args.output_options:
                    draw_args['fontsize'] =\
                            cl_args.output_options['fontsize']
                if is_pregroup_diagram(diagram):
                    discopy.grammar.draw(diagram, **draw_args)
                else:
                    try:
                        normal_form = discopy.grammar.normal_form(diagram)
                        discopy.grammar.draw(normal_form,  # pragma: no cover
                                             **draw_args)
                    except ValueError:
                        diagram.draw(**draw_args)


def main() -> None:
    parser = prepare_parser()
    cl_args = parser.parse_args()
    all_modules = [FileReaderModule(),
                   ParserModule(),
                   RewriterModule(),
                   AnsatzModule()]
    all_modules += [DiagramSaveModule()] if cl_args.mode == "string-diagram" \
                                         else [CCGTreeSaveModule()]
    if cl_args.load_args is not None:
        saved_args = yaml.load(open(cl_args.load_args, 'r'),
                               Loader=yaml.FullLoader)
        for key, value in saved_args.items():
            if not hasattr(cl_args, key) or\
               getattr(cl_args, key) is None or\
               getattr(cl_args, key) is False or\
               getattr(cl_args, key) == '' or\
               getattr(cl_args, key) == {}:
                setattr(cl_args, key, value)
    for key, value in DEFAULT_ARG_VALUES.items():
        if not hasattr(cl_args, key) or\
           getattr(cl_args, key) is None or\
           getattr(cl_args, key) is False or\
           getattr(cl_args, key) == '' or\
           getattr(cl_args, key) == {}:
            setattr(cl_args, key, value)
    validate_args(cl_args)
    if cl_args.store_args is not None:
        with open(cl_args.store_args, 'w') as f:
            cl_args.store_args = None
            yaml.dump(vars(cl_args), f, default_flow_style=False)
    data = None
    for module in all_modules:
        data = module(cl_args, data)


if __name__ == '__main__':  # pragma: no cover
    main()
