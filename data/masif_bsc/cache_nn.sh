masif_root=/home/yanyuan.zhang/project/masif-toughm1/masif-BSC
masif_source=$masif_root/source
masif_data=$masif_root/data
masif_matlab=$masif_root/source/matlab_libs
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_ppi_search/

wd=$masif_root/data/masif_bsc
cd $wd
python3 $masif_source/masif_bsc/masif_bsc_cache_training_data.py
