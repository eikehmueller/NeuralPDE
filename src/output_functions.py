'''
This code moves output files from WSL into a folder in windows - primarily so that 
Paraview for windows can be used instead of Paraview for Linux (which is buggy).
'''

import os
import shutil
from firedrake import *

# Define your WSL and Windows folders
wsl_folder = '/home/katie795/internship/NeuralPDE/output'
windows_folder = 'C:\\Users\\kathe\\OneDrive\\Documents\\summer_internship\\paraview_data'



def move_files_and_directories(wsl_folder, windows_folder):
    """Only to be used on the laptop. Moves from WSL folder to a specified windows folder"""
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

def clear_output(folder):
    """Clears the contents of the output folder."""

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def write_to_vtk(V, name, dof_values, path_to_output, time=None):
    """Given an tensor, writes it to a vtk file"""
    u = Function(V, name=name)
    u.dat.data[:] = dof_values
    file = VTKFile(f"{path_to_output}/{name}.pvd")
    file.write(u, time=time) 





# Call the function
#######################################################################
# M A I N
#######################################################################
if __name__ == "__main__":
    move_files_and_directories(wsl_folder, windows_folder)
