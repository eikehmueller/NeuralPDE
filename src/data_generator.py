"""This script produces the data"""

import argparse

from neural_pde.datasets import SolidBodyRotationDataset
from firedrake import FunctionSpace, UnitIcosahedralSphereMesh

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--degree",
    type=int,
    action="store",
    help="polynomial degree",
    default=4,
)

parser.add_argument(
    "--nsamples",
    type=int,
    action="store",
    help="number of samples",
    default=32,
)

parser.add_argument(
    "--seed",
    type=int,
    action="store",
    help="seed for rng",
    default=123447,
)

parser.add_argument(
    "--omega",
    type=float,
    action="store",
    help="angular velocity",
    default=1.0,
)

parser.add_argument(
    "--tfinalmax",
    type=float,
    action="store",
    help="maximal final time",
    default=1.0,
)

parser.add_argument(
    "--nref",
    type=int,
    action="store",
    help="number of refinements of spherical mesh",
    default=2,
)

parser.add_argument(
    "--filename",
    type=str,
    action="store",
    help="name of file to save to",
    default="dataset.h5",
)

args, _ = parser.parse_known_args()

print(f"generating data in file {args.filename}")
print(f"  nref      = {args.nref}")
print(f"  nsamples  = {args.nsamples}")
print(f"  omega     = {args.omega}")
print(f"  tfinalmax = {args.tfinalmax}")
print(f"  seed      = {args.seed}")
print(f"  degree    = {args.degree}")

dataset = SolidBodyRotationDataset(
    args.nref, args.nsamples, args.omega, args.tfinalmax, args.degree, args.seed
)
dataset.generate()
dataset.save(args.filename)
