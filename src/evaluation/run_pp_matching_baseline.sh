#!/bin/bash

# Path to the pp_matching_baseline.py script
SCRIPT_PATH="pp_matching_baseline.py"

# Path to the selected_lig.txt file
LIGAND_FILE="selected_lig.txt"

# Path to baseline generated mols
POCKET2MOL_DIR="/home2/conghao001/protein2drug/Pocket2Mol/outputs_selected_pdb"
TARGET_DIFF_DIR="/home2/conghao001/e3_mol_design/targetdiff/scripts/outputs_selected_pdb"

# Loop through each ligand name in the selected_lig.txt file
while IFS= read -r ligand_name; do
    echo "Running pp_matching_baseline.py for ligand: $ligand_name"
    python "$SCRIPT_PATH" -r "$POCKET2MOL_DIR" --ligand_name "$ligand_name"
    python "$SCRIPT_PATH" -r "$TARGET_DIFF_DIR" --ligand_name "$ligand_name"
done < "$LIGAND_FILE"