#!/bin/bash
file="$@" # Get the file name
if [[ -f "$file" ]]; then # 
    base=$(basename "$file" .pdb) # basename is something like '1bbuA_pocket.pdb'. Remove .pdb
    base=${base/_pocket/}  # Remove _pocket
    ID_CHAIN=$(echo "$base" | sed -E 's/(.*)([A-Za-z])$/\1_\2/')   # Insert underscore before last uppercase
    echo $ID_CHAIN
    ./data_prepare_one.sh -f $file $ID_CHAIN
fi

