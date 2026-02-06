"""This script produces the data"""

import argparse
import os
import shutil
from neural_pde.data.datasets import (
    SolidBodyRotationDataset,
    ShallowWaterEquationsDataset,
)

# Create argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--PDE", type=str, action="store", help="The PDE problem that is being solved"
)

parser.add_argument(
    "--degree",
    type=int,
    action="store",
    help="polynomial degree",
    default=4,
)

parser.add_argument(
    "--seed",
    type=int,
    action="store",
    help="seed for rng",
    default=123447,
)


parser.add_argument(
    "--filename",
    type=str,
    action="store",
    help="name of file to save to",
    default="../data/dataset.h5",
)

parser.add_argument(
    "--gusto_output_file_path",
    type=str,
    action="store",
    help="name of file where the gusto results are saved",
    default="",
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
    "--radius",
    type=float,
    action="store",
    help="radius of the sphere",
    default=1,
)

parser.add_argument(
    "--timescale",
    type=float,
    action="store",
    help="characteristic timescale of the simulation",
    default=1,
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
    "--t_lowest",
    type=float,
    action="store",
    help="start of time simulation",
    default=0,
)

parser.add_argument(
    "--t_highest", type=float, action="store", help="end of time simulation", default=10
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
    "--nsamples",
    type=int,
    action="store",
    help="number of samples",
    default=32,
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

args, _ = parser.parse_known_args()

print(f"  PDE                    = {args.PDE}")
print(f"  degree                 = {args.degree}")
print(f"  seed                   = {args.seed}")
print(f"  filename               = {args.filename}")
print(f"  gusto_output_file_path = {args.gusto_output_file_path}")
print(f"  nref                   = {args.nref}")
print(f"  omega                  = {args.omega}")
print(f"  g                      = {args.g}")
print(f"  radius                 = {args.radius}")
print(f"  timescale              = {args.timescale}")
print(f"  dt                     = {args.dt}")
print(f"  tfinalmax              = {args.tfinalmax}")
print(f"  t_lowest               = {args.t_lowest}")
print(f"  t_highest              = {args.t_highest}")
print(f"  t_interval             = {args.t_interval}")
print(f"  t_sigma                = {args.t_sigma}")
print(f"  nsamples               = {args.nsamples}")
print(f"  regenerate_data        = {args.regenerate_data}")
print(f"  save_diagnostics       = {args.save_diagnostics}")


if args.PDE == "SBR":
    print("Generating Solid Body Rotation dataset")
    dataset = SolidBodyRotationDataset(
        args.nref, args.nsamples, args.omega, args.tfinalmax, args.degree, args.seed
    )
    dataset.generate()
    dataset.save(args.filename)
elif args.PDE == "SWE":
    print("Generating Shallow Water Equations dataset")
    dataset = ShallowWaterEquationsDataset(
        n_ref=args.nref,
        nsamples=args.nsamples,
        dt=args.dt,
        t_final_max=args.tfinalmax,
        omega=args.omega,
        g=args.g,
        radius=args.radius,
        timescale=args.timescale,
        t_interval=args.t_interval,
        t_sigma=args.t_sigma,
        t_lowest=args.t_lowest,
        t_highest=args.t_highest,
    )

    if not os.path.isdir(args.gusto_output_file_path):
        print("Generating the full simulation")
        dataset.generate_full_dataset()
    elif args.regenerate_data:
        print("Regenerating the full simulation")
        shutil.rmtree(args.gusto_output_file_path)
        dataset.generate_full_dataset()
    else:
        print("Opening previously generated simulation")

    print("Extracting the data for the training, test, and validation sets")
    dataset.prepare_for_model(os.path.join(args.gusto_output_file_path, "chkpt.h5"))

    print("Saving the data in h5 format")
    dataset.save(args.filename)
else:
    print("PDE options are SBR (Solid Body Rotation) or SWE (Shallow Water Equations)")
