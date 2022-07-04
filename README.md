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
- If you want to subscribe to lambeq's mailing list, let us know by sending an email to <lambeq-support@cambridgequantum.com>. 

---
**Note:** Please do not try to read the documentation directly from the preview provided in the [repository](https://github.com/CQCL/lambeq/tree/main/docs), since some of the pages will not be rendered properly.

---

## Getting started

### Prerequisites

- Python 3.8+

### Installation

lambeq can be installed with the command:
```bash
pip install lambeq
```

The default installation of lambeq includes Bobcat parser, a state-of-the-art statistical parser (see [related paper](https://arxiv.org/abs/2109.10044)) fully integrated with the toolkit.

To install lambeq with optional dependencies for extra features, run:
```bash
pip install lambeq[extras]
```

To enable depccg support, you will need to install depccg separately. More information can be found
on the [depccg homepage](//github.com/masashi-y/depccg).
Currently, only version 2.0.3.2 of depccg is supported. After installing depccg, you can download its model by using the script provided in the `contrib` folder of this repository:

```bash
python contrib/download_depccg_model.py
```

## Usage

The [docs/examples](//github.com/CQCL/lambeq/tree/main/docs/examples)
directory contains notebooks demonstrating usage of the various tools in
lambeq.

Example - parsing a sentence into a diagram (see
[docs/examples/ccg2discocat.ipynb](//github.com/CQCL/lambeq/blob/main/docs/examples/ccg2discocat.ipynb)):

```python
from lambeq import BobcatParser

parser = BobcatParser()
diagram = parser.sentence2diagram('This is a test sentence')
diagram.draw()
```

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

## License

Distributed under the Apache 2.0 license. See [`LICENSE`](LICENSE) for
more details.

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
