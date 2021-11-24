# lambeq

[![lambeq logo](https://cqcl.github.io/lambeq/_static/lambeq_logo.png)](//cqcl.github.io/lambeq)

![Build status](https://github.com/CQCL/lambeq/actions/workflows/build_test.yml/badge.svg)
[![License](https://img.shields.io/github/license/CQCL/lambeq)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/lambeq)](//pypi.org/project/lambeq)
[![PyPI downloads](https://img.shields.io/pypi/dm/lambeq)](//pypi.org/project/lambeq)
[![arXiv](https://img.shields.io/badge/arXiv-2110.04236-green)](//arxiv.org/abs/2110.04236)

## About

lambeq is a toolkit for quantum natural language processing (QNLP).

- Documentation: https://cqcl.github.io/lambeq/
- User support: <lambeq-support@cambridgequantum.com>
- Contributions: Please read [our guide](https://cqcl.github.io/lambeq/CONTRIBUTING.html).
- If you want to subscribe to lambeq's mailing list, send an email to <lambeq-users@cambridgequantum.com> with the word "subscribe" as subject.

---
**Note:** Please do not try to read the documentation directly from the preview provided in the [repository](https://github.com/CQCL/lambeq/tree/main/docs), since some of the pages will not be rendered properly.

---

## Getting started

### Prerequisites

- Python 3.7+

### Installation

#### Direct pip install

The base lambeq can be installed with the command:
```bash
pip install lambeq
```

This does not include optional dependencies such as depccg and PyTorch,
which have to be installed separately. In particular, depccg is required
for `lambeq.ccg2discocat.DepCCGParser`.

---
**Warning:** depccg is available only on MacOS and Linux. If you are using Windows, please install the base lambeq. This means that the `DepCCGParser` class will not be available on Windows, but you can still use all other compositional models from the `reader` module. Support for parsing on Windows will be added in a future version.

---

To install lambeq with depccg, run instead:
```bash
pip install cython numpy
pip install 'lambeq[depccg]'
depccg_en download
```
See below for further options.

#### Automatic installation (recommended)

This runs an interactive installer to help pick the installation
destination and configuration.

1. Run:
   ```bash
   sh <(curl 'https://cqcl.github.io/lambeq/install.sh')
   ```

#### Git installation

This requires Git to be installed.

1. Download this repository:
   ```bash
   git clone https://github.com/CQCL/lambeq
   ```

2. Enter the repository:
   ```bash
   cd lambeq
   ```

3. Make sure `pip` is up-to-date:

   ```bash
   pip install --upgrade pip wheel
   ```

4. (Optional) If installing the optional depccg dependency, the
   following packages must be installed *before* depccg:
   ```bash
   pip install cython numpy
   ```
   Further information can be found on the
   [depccg homepage](//github.com/masashi-y/depccg).

5. Install lambeq from the local repository using pip:
   ```bash
   pip install --use-feature=in-tree-build .
   ```

   To include depccg, run instead:
   ```bash
   pip install --use-feature=in-tree-build .[depccg]
   ```

   To include all optional dependencies, run instead:
   ```bash
   pip install --use-feature=in-tree-build .[all]
   ```

6. If using a pretrained depccg parser,
[download a pretrained model](//github.com/masashi-y/depccg#using-a-pretrained-english-parser):
   ```bash
   depccg_en download
   ```

## Usage

The [docs/examples](//github.com/CQCL/lambeq/tree/main/docs/examples)
directory contains notebooks demonstrating usage of the various tools in
lambeq.

Example - parsing a sentence into a diagram (see
[docs/examples/ccg2discocat.ipynb](//github.com/CQCL/lambeq/blob/main/docs/examples/ccg2discocat.ipynb)):

```python
from lambeq.ccg2discocat import DepCCGParser

depccg_parser = DepCCGParser()
diagram = depccg_parser.sentence2diagram('This is a test sentence')
diagram.draw()
```

Note: all pre-trained depccg models apart from the basic one are broken,
and depccg has not yet been updated to fix this. Therefore, it is
recommended to just use the basic parser, as shown here.

## Testing

Run all tests with the command:

```bash
pytest
```

Note: if you have installed in a virtual environment, remember to
install pytest in the same environment using pip.

## Building documentation

To build the documentation, first install the required dependencies:

```bash
pip install -r docs/requirements.txt
```
then run the commands:

```bash
cd docs
make clean
make html
```
the docs will be under `docs/_build`.

To rebuild the rst files themselves, run:

```bash
sphinx-apidoc --force -o docs lambeq
```

## License

Distributed under the Apache 2.0 license. See [`LICENSE`](LICENSE) for more details.

## Citation

If you wish to attribute our work, please cite
[the accompanying paper](//arxiv.org/abs/2110.04236):

```
@article{kartsaklis2021lambeq,
   title={lambeq: {A}n {E}fficient {H}igh-{L}evel {P}ython {L}ibrary for {Q}uantum {NLP}},
   author={Dimitri Kartsaklis and Ian Fan and Richie Yeung and Anna Pearson and Robin Lorenz and Alexis Toumi and Giovanni de Felice and Konstantinos Meichanetzidis and Stephen Clark and Bob Coecke},
   year={2021},
   journal={arXiv preprint arXiv:2110.04236},
}
```
