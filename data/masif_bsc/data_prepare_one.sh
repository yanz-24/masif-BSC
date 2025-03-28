#!/bin/bash
masif_root=/home/yanyuan.zhang/project/masif-toughm1/masif-BSC
masif_source=$masif_root/source
masif_matlab=$masif_root/source/matlab_libs
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

wd=$masif_root/data/masif_bsc
cd $wd


if [ "$1" == "--file" ] || [ "$1" == "-f" ]; then
	# check if input file provided
	if [ -z "$2" ]; then
		printf "[ERROR] Usage: %s [-f|--file <file_path>] <PDBID_CHAIN>\n" "$0"
		exit 1
	fi
	
	printf "[INFO] Running masif on user's customized file: %s\n" "$2"
	FILENAME=$2
	PDBID_CHAIN=$3

	# check if the file has been precomputed
	output_dir_04=$wd/data_preparation/04a-precomputation_9A/precomputation
	output_04=$output_dir_04/$PDBID_CHAIN

	echo $output_04
	if [ -e "$output_04" ]; then
		printf "[INFO] Precomputed file %s already exists. Skipping precomputation.\n" "$output_04"
		exit 0
	fi
	
	PDBID=$(echo $PDBID_CHAIN| cut -d"_" -f1)
	CHAIN=$(echo $PDBID_CHAIN| cut -d"_" -f2)

	if [ ! -f "$FILENAME" ]; then
		printf "[ERROR] File %s does not exist.\n" "$FILENAME"
		exit 1
	fi
	
	output_dir_00=$wd/data_preparation/00-raw_pdbs
	mkdir -p $output_dir_00
	cp $FILENAME $output_dir_00/$PDBID\.pdb
	
	
else
	# download from PDB
	printf "[INFO] Downloading %s from PDB\n" "$1"
	PDBID_CHAIN=$1
	PDBID=$(echo $PDBID_CHAIN| cut -d"_" -f1)
	CHAIN=$(echo $PDBID_CHAIN| cut -d"_" -f2)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $PDBID_CHAIN
fi

printf "[INFO] 01 Extracting and triangulating %s\n" "$PDBID_CHAIN"
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDBID\_$CHAIN

printf "[INFO] 04 Computing descriptors for %s\n" "$PDBID_CHAIN"
python $masif_source/data_preparation/04-masif_precompute.py masif_site $PDBID_CHAIN
