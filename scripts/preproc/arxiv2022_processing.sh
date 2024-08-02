#!/bin/bash
python ./preprocessing/arxiv_processing/unarxiv_title_ext.py \
    --data_dir ./data/STRoGeNS-arXiv22/rawdata \
    --output_dir ./data/STRoGeNS-arXiv22/rw\
    --log_dir ./logs/STRoGeNS-arXiv22/rw &

sleep 10m

python ./preprocessing/arxiv_processing/unarxiv_add_ref_info.py \
    --data_dir ./data/STRoGeNS-arXiv22/rw \
    --output_dir ./data/STRoGeNS-arXiv22/rw_wabst \
    --log_dir ./logs/STRoGeNS-arXiv22/rw_wabst

python ./preprocessing/arxiv_processing/unarxiv_convert2hg_format.py \
        --data_dir ./data/STRoGeNS-arXiv22/rw_wabst \
        --output_dir ./data/STRoGeNS-arXiv22/hg_format 
