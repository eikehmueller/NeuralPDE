"""Show header of a hdf5 file"""

import argparse
from neural_pde.data_classes import show_hdf5_header

parser = argparse.ArgumentParser()

parser.add_argument(
    "--filename",
    type=str,
    action="store",
    help="file to inspect",
    required=True,
)

args, _ = parser.parse_known_args()

show_hdf5_header(args.filename)
