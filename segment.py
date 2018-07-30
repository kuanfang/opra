#!/usr/bin/env python

"""Segment videos in the OPRA Dataset according to the annotations.

Author: Te-Lin Wu, Kuan Fang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--annotations',
            dest='annotations',
            help='Path of the annotations.',
            type=str,
            required=True)

    parser.add_argument(
            '--raw',
            dest='raw_video_dir',
            help='Directory of raw videos',
            type=str,
            default='playlists')

    parser.add_argument(
            '--output',
            dest='output_dir',
            help='The output directory.',
            type=str,
            default='clips')

    args = parser.parse_args()

    return args


def read_annotations(filename, num_points=10):
    """Read the annotations from file.

    Args:
        filename: Path to the annotations.
        num_points: Number of annotated points.

    Returns:
        annotations: The dict of annotations.
    """
    annotations = []

    with open(filename, 'r') as fin:
        for line in fin:
            items = line.split(' ')

            points = []
            for i in xrange(num_points):
                x = float(items[8 + 2*i])
                y = float(items[8 + 2*i + 1])
                points.append([x, y])

            entry = {
                    'channel': items[0],
                    'playlist': items[1],
                    'video': items[2],
                    'start_time': items[3],
                    'duration': items[4],
                    'image': items[5],
                    'image_shape': (float(items[6]), float(items[7])),
                    'points': points,
                    }
            annotations.append(entry)

    return annotations


def segment_videos(annotations, raw_video_dir, output_dir):
    """Segment the videos.

    Args:
        annotations: The dict of the annotations.
        raw_video_dir: Directory of raw videos.
    
    Returns:
        output_dir: Output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_annotations = len(annotations)
    for i, entry in enumerate(annotations):
        input_path = os.path.join(
                raw_video_dir, entry['channel'], entry['playlist'],
                '%s.mp4' % (entry['video']))
        video_output_dir = os.path.join(output_dir, entry['channel'],
                                        entry['playlist'], entry['video'])
        output_path = os.path.join(
                video_output_dir,
                '%s_%s.mp4' % (entry['start_time'], entry['duration']))

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        command = ('ffmpeg -y -ss {} -i {} -t {} -c copy {}').format(
                entry['start_time'], input_path, entry['duration'], output_path)
        print('Segmenting video (%d / %d) by calling [%s] ...' %
              (i, num_annotations, command))
        os.system(command)


def main():
    args = parse_args()

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    annotations = read_annotations(args.annotations)
    segment_videos(annotations, args.raw_video_dir, args.output_dir)


if __name__ == '__main__':
    main()
