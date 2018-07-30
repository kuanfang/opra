#!/bin/bash

$DATA_DIR='/capri16/kuanfang/datasets/opra'

python download.py --playlist '${DATA_DIR}/playlists' --outputs '${DATA_DIR}/raw_videos/'
python segment.py --annotations '${DATA_DIR}/train.txt' --raw '${DATA_DIR}/raw_videos/' --output '${DATA_DIR}/clips'
python segment.py --annotations '${DATA_DIR}/test.txt' --raw '${DATA_DIR}/raw_videos/' --output '${DATA_DIR}/clips'
