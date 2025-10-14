"""Visualise some parts of the dataset from the hdf5 file"""

from torch.utils.data import DataLoader
from firedrake import *
import os
import shutil
import argparse

from neural_pde.datasets import load_hdf5_dataset, show_hdf5_header

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    type=str,
    action="store",
    help="path to output folder",
    default="output_for_visualisation",
)

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="file containing the data",
    default="data/data_valid_swes_nref_3_0.0001.h5",
)

args, _ = parser.parse_known_args()

print()
print(f"==== data ====")
print()

show_hdf5_header(args.data)
print()

dataset = load_hdf5_dataset(args.data)

batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

if not os.path.exists(args.output):
    os.makedirs(args.output)

mesh = UnitIcosahedralSphereMesh(dataset.n_ref)
V = FunctionSpace(mesh, "CG", 1)
for j, ((X, t), y_target) in enumerate(iter(dataset)):

    print(t)

    f_input_d = Function(V, name="input_d")
    f_input_d.dat.data[:] = X.detach().numpy()[3, :]
    f_input_u1 = Function(V, name="input_u1")
    f_input_u1.dat.data[:] = X.detach().numpy()[4, :]
    f_input_u2 = Function(V, name="input_u2")
    f_input_u2.dat.data[:] = X.detach().numpy()[5, :]
    f_input_u3 = Function(V, name="input_u3")
    f_input_u3.dat.data[:] = X.detach().numpy()[6, :]

    f_target_d = Function(V, name="target_d")
    f_target_d.dat.data[:] = y_target.detach().numpy()[0, :]
    f_target_u1 = Function(V, name="target_u1")
    f_target_u1.dat.data[:] = y_target.detach().numpy()[1, :]
    f_target_u2 = Function(V, name="target_u2")
    f_target_u2.dat.data[:] = y_target.detach().numpy()[2, :]
    f_target_u3 = Function(V, name="target_u3")
    f_target_u3.dat.data[:] = y_target.detach().numpy()[3, :]

    file = VTKFile(os.path.join(args.output, f"output_{j:04d}.pvd"))
    file.write(f_input_d, f_input_u1, f_input_u2, f_input_u3, f_target_d, f_target_u1, f_target_u2, f_target_u3)



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

#move_files_and_directories(wsl_folder, windows_folder)