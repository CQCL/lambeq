from pathlib import Path
from itertools import chain
import nbformat as nbf


print("Cleaning notebooks...")

nbs_path = Path("examples")
tut_path = Path("tutorials")
useful_metadata = ["nbsphinx", "raw_mimetype"]

for file in chain(nbs_path.iterdir(), tut_path.iterdir()):
    if not (file.is_file() and file.suffix == ".ipynb"):
        continue

    ntbk = nbf.read(file, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        # Delete cell ID if it's there
        cell.pop("id", None)

        # Keep only useful metadata
        new_metadata = {x: cell.metadata[x]
                        for x in useful_metadata
                        if x in cell.metadata}
        cell.metadata = new_metadata

    ntbk.metadata = {"language_info": {"name": "python"}}

    # We need the version of nbformat to be x.4, otherwise cells IDs
    # are regenerated automatically
    ntbk.nbformat = 4
    ntbk.nbformat_minor = 4

    nbf.write(ntbk, file, version=nbf.NO_CONVERT)
