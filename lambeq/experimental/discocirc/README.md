# DisCoCirc extension for lambeq

Functionality to convert text into DisCoCirc string diagrams, using lambeq's grammar backend.

## Installation

Installing the experimental subpackage requires Python 3.10.

```bash
git clone git@github.com:CQCL/lambeq.git
cd lambeq
pip install ".[experimental]"
```

## Usage

To get DisCoCirc diagrams using frames:

```python
from lambeq.experimental.discocirc import DisCoCircReader

reader = DisCoCircReader()
reader.text2circuit('Alice likes Bob. Bob likes Alice too.').draw()
```

To get DisCoCirc diagrams with frames decomposed into multiple boxes:

```python
reader.text2circuit('Alice likes Bob. Bob likes Alice too.', sandwich=True).draw()
```
