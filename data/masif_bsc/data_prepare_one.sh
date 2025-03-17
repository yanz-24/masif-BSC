#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source

if [ "$1" == "--file" ] || [ "$1" == "-f" ]
then
	echo "Running masif site on user's customized file $2"
	ID_CHAIN=$3
	PDB_ID=$(echo $ID_CHAIN| cut -d"_" -f1)
	CHAIN=$(echo $ID_CHAIN| cut -d"_" -f2)
	FILENAME=$2
	mkdir -p data_preparation/00-raw_pdbs/
	cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID\.pdb
else
	# download from PDB
	echo "Downloading $2 from PDB"
	$ID_CHAIN=$1
	PDB_ID=$(echo $PPI_PAIR_ID| cut -d"_" -f1)
	CHAIN=$(echo $PPI_PAIR_ID| cut -d"_" -f2)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $ID_CHAIN
fi

python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN
python $masif_source/data_preparation/04-masif_precompute.py masif_site $ID_CHAIN
