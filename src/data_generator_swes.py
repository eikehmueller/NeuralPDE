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
    "--nt",
    type=int,
    action="store",
    help="number of time-steps in full dataset",
    default=10,
)

parser.add_argument(
    "--tfinalmax",
    type=float,
    action="store",
    help="maximal final time",
    default=0.0001,
)

parser.add_argument(
    "--nsamples",
    type=int,
    action="store",
    help="number of samples",
    default=32,
)

parser.add_argument(
    "--move_to_windows",
    type=bool,
    action="store",
    help="whether to move paraview data to windows folder",
    default=False,
)

args, _ = parser.parse_known_args()

print(f"  filename  = {args.filename}")
print(f"  nref      = {args.nref}")
print(f"  omega     = {args.omega}")
print(f"  g         = {args.g}")
print(f"  nt        = {args.nt}")
print(f"  tfinalmax = {args.tfinalmax}")
print(f"  nsamples  = {args.nsamples}")


dataset = ShallowWaterEquationsDataset(
    n_ref=args.nref, nsamples=args.nsamples, nt=args.nt, t_final_max=args.tfinalmax,
    omega=args.omega, g=args.g
)

if not os.path.isdir('/home/katie795/NeuralPDE_workspace/NeuralPDE/src/results/output'):
    print('Generating the data')
    dataset.generate_full_dataset()

print('Extracting the data for the training, test, and validation sets')
dataset.prepare_for_model()
dataset.save(args.filename)

def move_files_and_directories(wsl_folder, windows_folder):
    # Convert the Windows folder path to a format that WSL understands
    windows_folder_in_wsl = f'/mnt/{windows_folder[0].lower()}' + windows_folder[2:].replace('\\', '/')
    
    # Ensure the target directory exists
    if not os.path.exists(windows_folder_in_wsl):
        os.makedirs(windows_folder_in_wsl)
    
    # Move each file and directory from WSL folder to Windows folder
    for item in os.listdir(wsl_folder):
        wsl_path = os.path.join(wsl_folder, item)
        windows_path = os.path.join(windows_folder_in_wsl, item)
        
        # Move the file or directory
        shutil.move(wsl_path, windows_path)
        print(f'Moved: {wsl_path} -> {windows_path}')

# Define your WSL and Windows folders
wsl_folder = '/home/katie795/NeuralPDE_workspace/NeuralPDE/src/results/output/field_output'
windows_folder = 'C:\\Users\\kathe\\OneDrive\\Desktop\\paraview_data'

if args.move_to_windows:
    move_files_and_directories(wsl_folder, windows_folder)