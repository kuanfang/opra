#!/bin/bash

DATA_DIR='./data'

# Download and unzip the data folder.
wget ftp://cs.stanford.edu/cs/cvgl/OPRA/data.zip
unzip data.zip

# Download product review videos from YouTube.
python download.py --playlist ${DATA_DIR}'/playlists' --output ${DATA_DIR}'/raw_videos/'

# Segment the videos according to the annotations.
python segment.py --annotations ${DATA_DIR}'/annotations/train.txt' --raw ${DATA_DIR}'/raw_videos/' --output ${DATA_DIR}'/clips'
python segment.py --annotations ${DATA_DIR}'/annotations/test.txt' --raw ${DATA_DIR}'/raw_videos/' --output ${DATA_DIR}'/clips'
