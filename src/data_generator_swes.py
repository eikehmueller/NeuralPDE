"""This script produces the data"""
import sys
sys.path.append("/home/katie795/software/gusto")
import argparse

from neural_pde.datasets import ShallowWaterEquationsDataset

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--nsamples",
    type=int,
    action="store",
    help="number of samples",
    default=32,
)

parser.add_argument(
    "--t_final_max",
    type=float,
    action="store",
    help="maximal final time",
    default=0.0001,
)

parser.add_argument(
    "--nref",
    type=int,
    action="store",
    help="number of refinements of spherical mesh",
    default=2,
)

parser.add_argument(
    "--nt",
    type=int,
    action="store",
    help="number of time-steps in full dataset",
    default=2,
)

parser.add_argument(
    "--omega",
    type=float,
    action="store",
    help="angular velocity",
    default=7.292e-5,
)

parser.add_argument(
    "--g",
    type=float,
    action="store",
    help="gravitational force",
    default=9.80616,
)

parser.add_argument(
    "--filename",
    type=str,
    action="store",
    help="name of file to save to",
    default="data/data_swes.h5",
)

args, _ = parser.parse_known_args()

print(f"  omega     = {args.omega}")
print(f"  g         = {args.g}")
print(f"  nref      = {args.nref}")
print(f"  nsamples  = {args.nsamples}")
print(f"  nt        = {args.nt}")
print(f"  tfinalmax = {args.t_final_max}")
print(f"  filename  = {args.filename}")


dataset = ShallowWaterEquationsDataset(
    n_ref=args.nref, nsamples=args.nsamples, nt=args.nt, t_final_max=args.t_final_max,
    omega=args.omega, g=args.g
)
dataset.generate_full_dataset()
dataset.prepare_for_model()
dataset.save(args.filename)