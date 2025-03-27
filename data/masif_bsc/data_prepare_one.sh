#!/bin/bash
echo $1, $2, $3
masif_root=/home/yanyuan.zhang/project/masif-toughm1/masif-BSC
masif_source=$masif_root/source
wd=$masif_root/data/masif_bsc
cd $wd
export PYTHONPATH=$PYTHONPATH:$masif_source

if [ "$1" == "--file" ] || [ "$1" == "-f" ]; then
	# check if input file provided
	if [ -z "$2" ]; then
		echo "Usage: $0 [-f|--file <file_path>] <ID_CHAIN>"
		exit 1
	else
		echo "Running masif on user's customized file $2"
		ID_CHAIN=$3
		PDBID=$(echo $ID_CHAIN| cut -d"_" -f1)
		CHAIN=$(echo $ID_CHAIN| cut -d"_" -f2)
		FILENAME=$2
		output_dir=$wd/data_preparation/00-raw_pdbs/
		mkdir -p $output_dir
		cp $FILENAME $output_dir/$PDBID\.pdb
	fi
	
else
	# download from PDB
	echo "Downloading $2 from PDB"
	ID_CHAIN=$1
	PDBID=$(echo $ID_CHAIN| cut -d"_" -f1)
	CHAIN=$(echo $ID_CHAIN| cut -d"_" -f2)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $ID_CHAIN
fi

echo "01 Extracting and triangulating $ID_CHAIN"
python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDBID\_$CHAIN

echo "04 Computing descriptors for $ID_CHAIN"
python $masif_source/data_preparation/04-masif_precompute.py masif_site $ID_CHAIN
