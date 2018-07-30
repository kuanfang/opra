#!/bin/bash

DATA_DIR='/capri16/kuanfang/datasets/OPRA'

python download.py --playlist ${DATA_DIR}'/playlists' --output ${DATA_DIR}'/raw_videos/'
python segment.py --annotations ${DATA_DIR}'/annotations/train.txt' --raw ${DATA_DIR}'/raw_videos/' --output ${DATA_DIR}'/clips'
python segment.py --annotations ${DATA_DIR}'/annotations/test.txt' --raw ${DATA_DIR}'/raw_videos/' --output ${DATA_DIR}'/clips'
