#!/bin/bash

count=0

# loop over all the files in TOUGH_M1
for pdb_chain in /home/yanyuan.zhang/project/masif-toughm1/data/tough-m1/TOUGH-M1_dataset/*/*.pdb; do
	dir_name=$(dirname "$pdb_chain")
	base_name=$(basename "$pdb_chain" .pdb)
	echo $dir_name,$base_name

	#counter 
    	((count++))
	(( count >= 5 )) && break

done
