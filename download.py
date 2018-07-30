#!/usr/bin/env python

"""Download videos in the OPRA Dataset from YouTube.

Author: Te-Lin Wu, Kuan Fang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import glob
import os


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--playlist',
            dest='playlist_dir',
            help='The playlist directory.',
            type=str,
            default='playlists')

    parser.add_argument(
            '--output',
            dest='output_dir',
            help='The output directory.',
            type=str,
            default='/capri16/kuanfang/datasets/opra/raw_videos_1')

    args = parser.parse_args()

    return args


def read_playlist(filename):
    """Read the playlist from file.

    Args:
        filename: Path to the playlist file.

    Returns:
        playlist: A dict of the playlist.
    """
    playlist = []

    with open(filename) as fin:
        num_videos = 0
        for line in csv.reader(fin, delimiter=','):
            if 'http' not in line[0]:
                continue
            entry = {
                    'index': num_videos + 1,
                    'url': line[0],
                    'channel': line[1],
                    'title': line[2],
                    'description': line[3],
                    }
            playlist.append(entry)
            num_videos += 1

    return playlist


def download_videos_in_playlist(playlist, output_dir):
    """Download videos in the playlist.

    Args: 
        playlist: A dict of the playlist.
        output_dir: The directory to save the downloaded videos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_videos = len(playlist)

    for i, entry in enumerate(playlist):
        filename = str(entry['index'])
        output_path = os.path.join(output_dir, '%s.mp4' % (filename))
        command = 'youtube-dl -U {} -f mp4 -o {}'.format(
                entry['url'], output_path)
        print('Downloading video (%d / %d) from %s...' %
              (i, num_videos, entry['url']))
        os.system(command)


def main():
    args = parse_args()

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    for filename in glob.glob(os.path.join(args.playlist_dir, '*', '*.csv')):
        playlist = read_playlist(filename)
        print('Playlist %s contains %d videos.' % (filename, len(playlist)))

        items = filename.strip('.csv').split('/')
        output_dir = os.path.join(args.output_dir, items[-2], items[-1])
        download_videos_in_playlist(playlist, output_dir)


if __name__ == '__main__':
    main()
