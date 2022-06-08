import argparse
from io import StringIO
import pytest
import unittest.mock
from unittest.mock import patch

from lambeq import BobcatParser, VerbosityLevel, WebParser
from lambeq import cli
from lambeq.cli import ArgumentList
from lambeq.cli import main


@pytest.fixture
def sentence_input() -> str:
    return 'John likes Mary'


@pytest.fixture
def type_raising_input() -> str:
    return 'What Alice and Bob do not know'


@pytest.fixture
def unicode_sentence_output() -> str:
    return ('John    likes    Mary\n'
            '────  ─────────  ────\n'
            ' n    n.r·s·n.l   n\n'
            ' ╰─────╯  │  ╰────╯')


@pytest.fixture
def unicode_type_raising_output() -> str:
    return ('    What     Alice     and     Bob       do            not           know\n'
            '───────────  ─────  ─────────  ───  ───────────  ───────────────  ─────────\n'
            'n·n.l.l·s.l    n    n.r·n·n.l   n   n.r·s·s.l·n  s.r·n.r.r·n.r·s  n.r·s·n.l\n'
            '│   │    │     ╰─────╯  │  ╰────╯    │  │  │  ╰─╮─╯    │    │  │   │  │  │\n'
            '│   │    │              │            │  │  │  ╭─╰─╮    │    │  │   │  │  │\n'
            '│   │    │              │            │  │  ╰╮─╯   ╰─╮──╯    │  │   │  │  │\n'
            '│   │    │              │            │  │  ╭╰─╮   ╭─╰──╮    │  │   │  │  │\n'
            '│   │    │              │            │  ╰──╯  ╰─╮─╯    ╰─╮──╯  │   │  │  │\n'
            '│   │    │              │            │        ╭─╰─╮    ╭─╰──╮  │   │  │  │\n'
            '│   │    │              │            ╰────────╯   ╰─╮──╯    ╰╮─╯   │  │  │\n'
            '│   │    │              │                         ╭─╰──╮    ╭╰─╮   │  │  │\n'
            '│   │    │              ╰─────────────────────────╯    ╰─╮──╯  ╰───╯  │  │\n'
            '│   │    │                                             ╭─╰──╮         │  │\n'
            '│   │    ╰─────────────────────────────────────────────╯    ╰─────────╯  │\n'
            '│   ╰────────────────────────────────────────────────────────────────────╯')


@pytest.fixture
def ascii_sentence_output() -> str:
    return ('John    likes    Mary\n'
            '____  _________  ____\n'
            ' n    n.r s n.l   n\n'
           r' \_____/  |  \____/')


@pytest.fixture
def unicode_sentence_cups() -> str:
    return ('START   John  likes   Mary\n'
            '─────  ─────  ─────  ─────\n'
            '  s    s.r·s  s.r·s  s.r·s\n'
            '  ╰─────╯  ╰───╯  ╰───╯  │')


@pytest.fixture
def ascii_sentence_cups() -> str:
    return ('START   John  likes   Mary\n'
            '_____  _____  _____  _____\n'
            '  s    s.r s  s.r s  s.r s\n'
           r'  \_____/  \___/  \___/  |')


@pytest.fixture
def multi_sentence_input() -> str:
    return 'This is a sentence.\nThis is another one.'


shared_bobcat_parser = BobcatParser(verbose=VerbosityLevel.SUPPRESS.value)
parser_patch = {'depccg': lambda **kwargs: shared_bobcat_parser, 'bobcat': lambda **kwargs: shared_bobcat_parser}  # DepCCG can crash during online tests
parser_patch_uninitialised = {'depccg': BobcatParser, 'bobcat': BobcatParser}  # Only for tests where a new instance of the parser is needed


def test_file_io(sentence_input, unicode_sentence_output):
    with patch('sys.argv',
               ['lambeq', '-i', 'sentence.txt', '-o', 'output.txt']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open',
               unittest.mock.mock_open(read_data=sentence_input)) as m:
        main()
        m.assert_any_call('sentence.txt', 'r')
        m.assert_called_with('output.txt', 'w')
        handle = m()
        handle.read.assert_called_once()
        handle.write.assert_called_once_with(unicode_sentence_output)


def test_sentence_arg(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() == unicode_sentence_output


def test_root_cat(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', '-c', 'PP', 'NP', '-t', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch_uninitialised),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() != unicode_sentence_output


def test_type_raising(type_raising_input, unicode_type_raising_output):
    with patch('sys.argv', ['lambeq', type_raising_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() == unicode_type_raising_output


def test_sentence_parser_arg(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', '-p', 'depccg', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() == unicode_sentence_output


def test_sentence_ascii(sentence_input, ascii_sentence_output):
    with patch('sys.argv', ['lambeq', '-f', 'text-ascii', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() == ascii_sentence_output


def test_pickle(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', '-i', 'sentence.txt',
                            '-f', 'pickle', '-o', 'output.pickle']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open',
               unittest.mock.mock_open(read_data=sentence_input)) as m:
        main()
        m.assert_any_call('sentence.txt', 'r')
        m.assert_called_with('output.pickle', 'wb')
        handle = m()
        handle.read.assert_called_once()
        handle.write.assert_called_once()


def test_json(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', '-i', 'sentence.txt',
                            '-f', 'json', '-o', 'output.json']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open',
               unittest.mock.mock_open(read_data=sentence_input)) as m:
        main()
        m.assert_any_call('sentence.txt', 'r')
        m.assert_called_with('output.json', 'w')
        handle = m()
        handle.read.assert_called_once()
        handle.write.assert_called()


def test_folder_creation(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-i', 'sentences.txt', '-t', '-d',
                            'image_folder', '-f', 'image', '-g', 'png']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open', unittest.mock.mock_open(
                                     read_data=multi_sentence_input)) as m,\
         patch('lambeq.cli.discopy.grammar.draw',
               new=unittest.mock.MagicMock()) as d,\
         patch('lambeq.cli.Path', new=unittest.mock.MagicMock()) as p:
        main()
        m.assert_any_call('sentences.txt', 'r')
        handle = m()
        handle.read.assert_called_once()
        p.assert_called_once()
        d.assert_called()


def test_image_args(sentence_input):
    with patch('sys.argv', ['lambeq', '-f', 'image', '-u', 'fig_width=16',
                            'fig_height=3', 'fontsize=12', '-o',
                            'diagram.pdf', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.discopy.grammar.draw',
               new=unittest.mock.MagicMock()) as d:
        main()
        d.assert_called()


def test_split_stdin_and_multisentece_image_error(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-s', '-f', 'image', '-u',
                            'fig_width=16', 'fig_height=3', 'fontsize=12',
                            '-o', 'diagram.png']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdin',
               new=StringIO(multi_sentence_input.replace('\n', ' '))),\
         patch('sys.stdout', new=StringIO()) as fake_out,\
         patch('lambeq.cli.discopy.grammar.draw',
               new=unittest.mock.MagicMock()) as d:
        with pytest.raises(ValueError):
            main()
        assert fake_out.getvalue().rstrip() == 'Text:'
        d.assert_not_called()


def test_reader(sentence_input, unicode_sentence_cups):
    with patch('sys.argv', ['lambeq', '-r', 'cups', sentence_input]),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out:
        main()
        assert fake_out.getvalue().rstrip() == unicode_sentence_cups


def test_tree_reader(sentence_input):
    with patch('sys.argv', ['lambeq', '-r', 'tree', '-f', 'image', '-u',
                            'fig_width=16', 'fig_height=3', 'fontsize=12',
                            '-o', 'diagram.pdf', sentence_input]),\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d:
        main()
        d.assert_called()


def test_stairs_reader(sentence_input):
    with patch('sys.argv', ['lambeq', '-r', 'stairs', '-f', 'image', '-u',
                            'fig_width=16', 'fig_height=3', 'fontsize=12',
                            '-o', 'diagram.pdf', sentence_input]),\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d:
        main()
        d.assert_called()


def test_IQP_ansatz_and_rewrites(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-i', 'sentences.txt', '-t',
                            '-d', 'image_folder', '-f', 'image', '-g', 'png',
                            '-w', 'prepositional_phrase', 'determiner',
                            '-a', 'iqp', '-n', 'dim_n=1', 'dim_s=1',
                            'n_layers=2']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open', unittest.mock.mock_open(
                                  read_data=multi_sentence_input)) as m,\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d,\
         patch('lambeq.cli.Path', new=unittest.mock.MagicMock()) as p:
        main()
        m.assert_any_call('sentences.txt', 'r')
        handle = m()
        handle.read.assert_called_once()
        p.assert_called_once()
        d.assert_called()


def test_spiders_ansatz(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-i', 'sentences.txt', '-t', '-d',
                            'image_folder', '-f', 'image', '-g', 'png',
                            '-a', 'spider', '-n', 'dim_n=3', 'dim_s=3',
                            'max_order=3']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open', unittest.mock.mock_open(
                                    read_data=multi_sentence_input)) as m,\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d,\
         patch('lambeq.cli.Path', new=unittest.mock.MagicMock()) as p:
        main()
        m.assert_any_call('sentences.txt', 'r')
        handle = m()
        handle.read.assert_called_once()
        p.assert_called_once()
        d.assert_called()


def test_mps_ansatz(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-i', 'sentences.txt', '-t',
                            '-d', 'image_folder', '-f', 'image', '-g', 'png',
                            '-a', 'mps', '-n', 'dim_n=3', 'dim_s=3',
                            'bond_dim=3', 'max_order=3']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open', unittest.mock.mock_open(
                                    read_data=multi_sentence_input)) as m,\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d,\
         patch('lambeq.cli.Path', new=unittest.mock.MagicMock()) as p:
        main()
        m.assert_any_call('sentences.txt', 'r')
        handle = m()
        handle.read.assert_called_once()
        p.assert_called_once()
        d.assert_called()


def test_tensor_ansatz(multi_sentence_input):
    with patch('sys.argv', ['lambeq', '-i', 'sentences.txt', '-t', '-d',
                            'image_folder', '-f', 'image', '-g', 'png',
                            '-a', 'tensor', '-n', 'dim_n=3', 'dim_s=3']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('lambeq.cli.open', unittest.mock.mock_open(
                                    read_data=multi_sentence_input)) as m,\
         patch('lambeq.cli.discopy.monoidal.Diagram.draw',
               new=unittest.mock.MagicMock()) as d,\
         patch('lambeq.cli.Path', new=unittest.mock.MagicMock()) as p:
        main()
        m.assert_any_call('sentences.txt', 'r')
        handle = m()
        handle.read.assert_called_once()
        p.assert_called_once()
        d.assert_called()


def test_arg_storing(sentence_input, unicode_sentence_output):
    with patch('sys.argv', ['lambeq', '-i', 'sentence.txt',
                            '-y', 'conf.yaml']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out,\
         patch('lambeq.cli.open',
               unittest.mock.mock_open(read_data=sentence_input)) as m:
        main()
        m.assert_any_call('sentence.txt', 'r')
        m.assert_any_call('conf.yaml', 'w')
        handle = m()
        handle.read.assert_called_once()
        handle.write.assert_called()
        assert fake_out.getvalue().rstrip() == unicode_sentence_output


def test_arg_loading(sentence_input, ascii_sentence_cups):
    yaml_input = (f'input_sentence: {sentence_input}\n'
                  'output_format: text-ascii\nreader: spiders\n')
    with patch('sys.argv', ['lambeq', '-r', 'cups', '-l', 'conf.yaml']),\
         patch('lambeq.cli.AVAILABLE_PARSERS', new=parser_patch),\
         patch('sys.stdout', new=StringIO()) as fake_out,\
         patch('lambeq.cli.open',
               unittest.mock.mock_open(read_data=yaml_input)) as m:
        main()
        m.assert_any_call('conf.yaml', 'r')
        handle = m()
        handle.read.assert_called()
        assert fake_out.getvalue().rstrip() == ascii_sentence_cups


@pytest.fixture
def arg_parser() -> argparse.ArgumentParser:
    return cli.prepare_parser()


def test_args_validation_two_inputs(arg_parser):
    cli_args = arg_parser.parse_args(['--input_file', 'dummy.txt',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_missing_output_file(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'pickle',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_both_parser_and_reader_given(arg_parser):
    cli_args = arg_parser.parse_args(['--parser', 'bobcat',
                                      '--reader', 'cups',
                                      '--output_format','text-unicode',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_redundant_output_args(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'text-unicode',
                                      '--output_options', 'fontsize=12',
                                      '-t', 'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_dir(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'text-unicode',
                                      '--output_dir', 'dummy_folder',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_ansatz(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'text-unicode',
                                      '--output_file', 'dummy_file',
                                      '--ansatz', 'iqp',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_rewrite(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'text-ascii',
                                      '--rewrite_rules', 'determiner',
                                      '--output_file', 'dummy_file',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_reader(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'text-ascii',
                                      '--output_file', 'dummy_file',
                                      '--reader', 'spiders',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_ccg_invalid_format_image(arg_parser):
    cli_args = arg_parser.parse_args(['--mode', 'ccg',
                                      '--output_format', 'image',
                                      '--output_file', 'dummy_file',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_ccg_invalid_output_reader(arg_parser):
    cli_args = arg_parser.parse_args(['--mode', 'ccg',
                                      '--output_file', 'dummy_file',
                                      '--reader', 'spiders',
                                      'Input sentence.'])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_correct_arglist_parsing(arg_parser):
    cli_args = arg_parser.parse_args(['--output_format', 'image',
                                      '--output_options', 'fig_width=15',
                                      'fig_height=5', 'fontsize=3',
                                      '--output_file', 'dummy_file',
                                      'Input sentence.'])
    assert cli_args.output_options['fig_width'] == 15
    assert cli_args.output_options['fig_height'] == 5
    assert cli_args.output_options['fontsize'] == 3


@pytest.fixture
def arg_list() -> ArgumentList:
    return ArgumentList([('test_int', int, 4), ('test_str', str, None)])


def test_arglist_all_options(arg_list):
    assert arg_list.all_options() == ('test_int=<int> (default: 4), '
                                      'test_str=<str> (default: None)')


def test_arglist_generator(arg_list):
    all_options = ', '.join(x for x in arg_list)
    assert all_options == arg_list.all_options()


def test_type_mismatch():
    with pytest.raises(ValueError):
        _ = ArgumentList([['test_int', int, 4.3], ['test_str', str, 3]])


def test_access(arg_list):
    assert 'test_int=5' in arg_list
    assert 'test_str=text' in arg_list
    assert 'test_int=text' not in arg_list
    assert 'test_bool=True' not in arg_list
    assert arg_list('test_int=6') == ('test_int', 6)
