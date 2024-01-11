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
from argparse import ArgumentParser
import re


def main():
    parser = ArgumentParser(description="Clean notebooks.")
    parser.add_argument('-p', '--docs_path', default='./docs/',
                        help='Path to lambeq docs directory')
    parser.add_argument('-s', '--suppress_warnings', action='store_true',
                        help='Whether or not to suppress warnings')
    args = parser.parse_args()

    print("Cleaning notebooks...")

    nbs_path = Path(args.docs_path + "/" + "examples")
    tut_path = Path(args.docs_path + "/" + "tutorials")
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

                    if args.suppress_warnings:
                        # Remove warnings
                        indices_to_remove = []
                        for idx, output in enumerate(cell.outputs):
                            if output.output_type == 'stream' and output.name == 'stderr':
                                stderr_text = output.text
                                warning_pattern = r'warnings\.warn\('
                                if re.search(warning_pattern, stderr_text):
                                    indices_to_remove.append(idx)

                        # Remove the identified entries from the outputs
                        # list in reverse order
                        for idx in reversed(indices_to_remove):
                            del cell.outputs[idx]

        ntbk.metadata = {"language_info": {"name": "python"}}

        # We need the version of nbformat to be x.4, otherwise cells IDs
        # are regenerated automatically
        ntbk.nbformat = 4
        ntbk.nbformat_minor = 4

        nbf.write(ntbk, file, version=nbf.NO_CONVERT)


if __name__ == "__main__":
    main()
