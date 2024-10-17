'''
This code moves output files from WSL into a folder in windows - primarily so that 
Paraview for windows can be used instead of Paraview for Linux (which is buggy).
'''

import os
import shutil
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

# Define your WSL and Windows folders
wsl_folder = '/home/katie795/internship/NeuralPDE/output'
windows_folder = 'C:\\Users\\kathe\\OneDrive\\Documents\\summer_internship\\paraview_data'


def move_files_and_directories(wsl_folder, windows_folder):
    """This function is used when working in WSL in windows. 
    Moves files from a WSL folder to a windows folder

    :arg wsl_folder: the path to the wsl folder
    :arg windows_folder: the path to the windows folder
    """
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
    """Clears the contents of the output folder.

    :arg folder: the path to the output folder"""

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
    """Given an numpy array, writes it to a vtk file.

    :arg V: the firedrake function space
    :arg name: string of chosen name for the function
    :arg dof_values: the numpy array with dof values on the vertices of the mesh
    :arg path_to_output: the path to the output folder
    :arg time (optional): the time in a time dependent simulation"""
    u = Function(V, name=name) # define the function
    u.dat.data[:] = dof_values # set dofs
    file = VTKFile(f"{path_to_output}/{name}.pvd") 
    file.write(u, time=time) 
    return


def training_plots(training_loss, training_loss_per_epoch, validation_loss_per_epoch, path_to_output, test_number):
    """Draw matplotlib diagrams from the data gathered in the training loop
    
    :arg training_loss: list of values in the training loss
    :arg training_loss_per_epoch: list of values in the training loss per epoch
    :arg validation loss per epoch: list of values for the validation loss per epoch
    :arg path_to_output: path to the output folder
    :arg test_number: test number
    """

    training_iterations = np.arange(0.0, len(training_loss), 1)
    epoch_iterations = np.arange(0.0, len(training_loss_per_epoch), 1)

    fig1, ax1 = plt.subplots()
    ax1.set_ylim([0, 1.1])
    ax1.plot(training_iterations, np.array(training_loss))
    ax1.set(xlabel='Number of training iterations', ylabel=r'Normalized $L^2$ loss',
            title='Training loss')
    ax1.grid()
    fig1.savefig(f'{path_to_output}/training_loss_test{test_number}.png')


    fig2, ax2 = plt.subplots()
    ax2.set_ylim([0, 1.1])
    ax2.plot(epoch_iterations, np.array(training_loss_per_epoch), label='Training loss', marker='o')
    ax2.plot(epoch_iterations, np.array(validation_loss_per_epoch), label='Validation loss', linestyle='dashed',
            marker='v')
    ax2.set(xlabel='Number of epochs', ylabel=r'Normalized $L^2$ loss',
            title='Training and validation loss per epoch')
    ax2.legend()
    ax2.grid()
    fig2.savefig(f'{path_to_output}/validation_loss_test{test_number}.png')

    fig3, ax3 = plt.subplots()
    ax1.set_ylim([0, 1.1])
    ax3.set_yscale('log')
    ax3.plot(epoch_iterations, np.array(training_loss_per_epoch), label='Training loss', marker='o')
    ax3.plot(epoch_iterations, np.array(validation_loss_per_epoch), label='Validation loss', linestyle='dashed',
            marker='v')
    ax3.set(xlabel='Number of epochs', ylabel=r'Normalized $L^2$ loss',
            title='Log of training and validation loss per epoch')
    ax3.legend()
    ax3.grid()
    fig3.tight_layout()
    fig3.savefig(f'{path_to_output}/log_loss{test_number}.png')
    plt.show()
    plt.close()


# Call the function
#######################################################################
# M A I N
#######################################################################
if __name__ == "__main__":
    move_files_and_directories(wsl_folder, windows_folder)
