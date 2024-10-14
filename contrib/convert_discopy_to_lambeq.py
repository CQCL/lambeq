"""
Script for converting DisCoPy circuits to lambeq circuits.

This requires `discopy>=1.1.7` to work.
"""
from argparse import ArgumentParser
import glob
import os
import pickle
import sys

import discopy
from lambeq.backend.quantum import Diagram as Circuit


MIN_DISCOPY_VERSION = '1.1.7'


def convert_discopy_to_lambeq(pkl_path: str) -> list[Circuit]:
    """Convert pickled DisCoPy circuits from disk to lambeq circuits.

    Parameters
    ----------
    pkl_path: path to pickle file or directory containing pickle files

    Returns
    -------
    List of lambeq circuits

    """

    if pkl_path.endswith(".pkl"):
        pkl_files = [pkl_path]
    else:
        pkl_files = glob.glob(pkl_path + "/*.pkl")

    lmbq_circs = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            dcp_circs = pickle.load(f)

        if not isinstance(dcp_circs, list):
            dcp_circs = [dcp_circs]

        for dcp_circ in dcp_circs:
            lmbq_circ = Circuit.from_discopy(dcp_circ)
            lmbq_circs.append(lmbq_circ)

    return lmbq_circs


def main():
    # Conversion is only supported from DisCoPy v1.1.7 and onwards.
    if not discopy.__version__ >= MIN_DISCOPY_VERSION:
        raise AssertionError(f'`discopy>={MIN_DISCOPY_VERSION}` is required by this script.')

    parser = ArgumentParser(description='Convert DisCoPy circuits to lambeq circuits')
    parser.add_argument('--discopy-circuit-path',
                        type=str,
                        help='Either a directory containing the pickled DisCoPy '
                             'circuits or a specific pickle file.',
                        default='discopy_circs')
    parser.add_argument('--output-dir',
                        type=str,
                        help='The directory where the output lambeq circuits '
                             'will be saved. The file will have the filename '
                             '`lambeq_circs.pkl`.',
                        default='output')
    args = parser.parse_args()

    # Create output dir if it doesn't exist yet
    os.makedirs(args.output_dir, exist_ok=True)

    lmbq_circs = convert_discopy_to_lambeq(args.discopy_circuit_path)
    pickle.dump(lmbq_circs,
                open(f'{args.output_dir}/lambeq_circs.pkl', 'wb'))


if  __name__ == "__main__":
    main()
