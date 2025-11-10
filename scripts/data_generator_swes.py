"""This script produces the data"""
import sys
sys.path.append("/home/katie795/software/gusto")
import argparse
import os
import shutil

from neural_pde.datasets import ShallowWaterEquationsDataset

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--filename",
    type=str,
    action="store",
    help="name of file to save to",
    default="data/data_swes.h5",
)

parser.add_argument(
    "--output_file_path",
    type=str,
    action="store",
    help="name of file where the gusto results are saved",
    default="/home/katie795/NeuralPDE_workspace/scripts/results/gusto_output",
)

parser.add_argument(
    "--regenerate_data",
    action="store_true",
    help="whether to overwrite the full simulation",
)

parser.add_argument(
    "--save_diagnostics",
    action="store_true",
    help="whether to save the diagnostic functions (div and vort)",
)

parser.add_argument(
    "--nref",
    type=int,
    action="store",
    help="number of refinements of spherical mesh",
    default=4,
)

parser.add_argument(
    "--omega",
    type=float,
    action="store",
    help="angular velocity 1/s",
    default=7.292e-5,
)

parser.add_argument(
    "--g",
    type=float,
    action="store",
    help="gravitational acceleration m/s^2",
    default=9.80616,
)

parser.add_argument(
    "--dt",
    type=float,
    action="store",
    help="length of time-steps in full dataset",
    default=0.1,
)

parser.add_argument(
    "--tfinalmax",
    type=float,
    action="store",
    help="maximal final time",
    default=1,
)

parser.add_argument(
    "--nsamples",
    type=int,
    action="store",
    help="number of samples",
    default=32,
)

parser.add_argument(
    "--t_interval",
    type=float,
    action="store",
    help="expected size of a time interval",
    default=1,
)

parser.add_argument(
    "--t_sigma",
    type=float,
    action="store",
    help="expected standard deviation of a time interval",
    default=0.5,
)

parser.add_argument(
    "--t_lowest",
    type=float,
    action="store",
    help="start of time simulation",
    default=0,
)

parser.add_argument(
    "--t_highest",
    type=float,
    action="store",
    help="end of time simulation",
    default=10
)


args, _ = parser.parse_known_args()

print(f"  filename         = {args.filename}")
print(f"  nref             = {args.nref}")
print(f"  omega            = {args.omega}")
print(f"  g                = {args.g}")
print(f"  nt               = {args.dt}")
print(f"  tfinalmax        = {args.tfinalmax}")
print(f"  t_lowest         = {args.t_lowest}")
print(f"  t_highest        = {args.t_highest}")
print(f"  nsamples         = {args.nsamples}")
print(f"  regenerate_data  = {args.regenerate_data}")
print(f"  output_file_path = {args.output_file_path}")

dataset = ShallowWaterEquationsDataset(
    n_ref=args.nref, nsamples=args.nsamples, dt=args.dt, t_final_max=args.tfinalmax,
    omega=args.omega, g=args.g, t_interval=args.t_interval, t_sigma=args.t_sigma,
    t_lowest=args.t_lowest, t_highest=args.t_highest
)

if not os.path.isdir(args.output_file_path):
    print('Generating the full simulation')
    dataset.generate_full_dataset()
elif args.regenerate_data:
    print('Regenerating the full simulation')
    shutil.rmtree(args.output_file_path)
    dataset.generate_full_dataset()
else:
    print('Opening previously generated simulation')

print('Extracting the data for the training, test, and validation sets')
dataset.prepare_for_model()

print('Saving the data in h5 format')
dataset.save(args.filename)
