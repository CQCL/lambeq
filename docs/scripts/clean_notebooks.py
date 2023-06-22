"""
This script performs the following actions on example and
tutorial notebooks:

- Removes cell IDs
- Keeps only `useful_metadata` for each cell
- Renumbers code cells, ignoring hidden ones
- Keeps only necessary notebook metadata
- Pins nbformat version
"""

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

    exec_count = 0

    for cell in ntbk.cells:
        # Delete cell ID if it's there
        cell.pop("id", None)
        if cell.get("attachments") == {}:
            cell.pop("attachments", None)

        # Keep only useful metadata
        new_metadata = {x: cell.metadata[x]
                        for x in useful_metadata
                        if x in cell.metadata}
        cell.metadata = new_metadata

        # Renumber execution counts, ignoring hidden cells
        if cell.cell_type == "code":
            if cell.metadata.get("nbsphinx") == "hidden":
                cell.execution_count = None
            else:
                exec_count += 1
                cell.execution_count = exec_count

            # Adjust the output execution count, if present
            if len(cell.outputs) > 0:
                output = cell.outputs[-1]  # execute_result must be
                                           # the last entry
                if output.output_type == "execute_result":
                    output.execution_count = cell.execution_count

    ntbk.metadata = {"language_info": {"name": "python"}}

    # We need the version of nbformat to be x.4, otherwise cells IDs
    # are regenerated automatically
    ntbk.nbformat = 4
    ntbk.nbformat_minor = 4

    nbf.write(ntbk, file, version=nbf.NO_CONVERT)
