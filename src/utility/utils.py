import os
from typing import List

def list_file_paths(folder_path:str) -> List[str]:
    """
    Generate the list of path for each file present in the folder
    """
    file_paths = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    return file_paths

def extract_policy_number(file_path:str) -> str:
    # Extract just the filename from the path
    filename = os.path.basename(file_path)
    
    # Remove the file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by underscore and take the last part
    parts = name_without_ext.split('_')
    if parts:
        return parts[-1]
    return None

