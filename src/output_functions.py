"""
Methods for output
"""

import os
import shutil
from firedrake import *


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
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    return


def write_to_vtk(V, name, dof_values, path_to_output, time=None):
    """Given an numpy array, writes it to a vtk file.

    :arg V: the firedrake function space
    :arg name: string of chosen name for the function
    :arg dof_values: the numpy array with dof values on the vertices of the mesh
    :arg path_to_output: the path to the output folder
    :arg time (optional): the time in a time dependent simulation"""
    u = Function(V, name=name)  # define the function
    u.dat.data[:] = dof_values  # set dofs
    file = VTKFile(f"{path_to_output}/{name}.pvd")
    file.write(u, time=time)
