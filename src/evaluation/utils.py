import os 

def build_pdb_dict(raw_data_path):
    pdb_dict = {}
    for directory in os.listdir(raw_data_path):
        dir_path = os.path.join(raw_data_path, directory)
        if os.path.isdir(dir_path):
            # Get the list of files in the directory
            files = os.listdir(dir_path)

            # Add the directory and its files to the dictionary
            pdb_dict[directory] = files
    pdb_rev_dict = {v:k for k, files in pdb_dict.items() for v in files}
    return pdb_dict, pdb_rev_dict