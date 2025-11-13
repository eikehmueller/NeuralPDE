import os
import shutil

### Move from wsl to windows for paraview ####
def move_files_and_directories(wsl_folder, windows_folder):
    # Convert the Windows folder path to a format that WSL understands
    windows_folder_in_wsl = f'/mnt/{windows_folder[0].lower()}' + windows_folder[2:].replace('\\', '/')
    
    # Ensure the target directory exists
    if not os.path.exists(windows_folder_in_wsl):
        os.makedirs(windows_folder_in_wsl)
    
    # Move each file and directory from WSL folder to Windows folder
    for item in os.listdir(wsl_folder):
        
        if item.endswith(".pvd"):
            pass # skip if it is a pvd file
        elif item.endswith(".vtu"):
            wsl_path = os.path.join(wsl_folder, item)
            windows_path = os.path.join(windows_folder_in_wsl, item)
            
            # Move the file or directory
            shutil.move(wsl_path, windows_path)
            print(f'Moved: {wsl_path} -> {windows_path}')
        else:
            new_wsl_folder = os.path.join(wsl_folder, item) # go into the folder 
            for subitem in os.listdir(new_wsl_folder): # extract vtu file from the folder
                wsl_path = os.path.join(new_wsl_folder, subitem)
                windows_path = os.path.join(windows_folder_in_wsl, subitem)
                
                # Move the file or directory
                shutil.move(wsl_path, windows_path)
                print(f'Moved: {wsl_path} -> {windows_path}')
