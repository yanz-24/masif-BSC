#!/bin/bash

# iterate over folder
for file in /home/yanyuan.zhang/project/masif-toughm1/data/pocket_6A_ligand/*; do
    if [[ -f "$file" ]]; then # 
        base=$(basename "$file" .pdb) # basename is something like '1bbuA_pocket.pdb'. Remove .pdb
        base=${base/_pocket/}  # Remove _pocket
        ID_CHAIN=$(echo "$base" | sed -E 's/(.*)([A-Za-z])$/\1_\2/')   # Insert underscore before last uppercase
        # save to file
        echo "$file,$ID_CHAIN" >> jobs.csv
    fi
done


