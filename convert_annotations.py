#!/usr/bin/env python

"""Segment videos in the OPRA Dataset according to the annotations.

This is used to convert the original annotations for data releasing.

Author: Te-Lin Wu, Kuan Fang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input',
            dest='input',
            help='The input filename',
            type=str,
            required=True)

    parser.add_argument(
            '--output',
            dest='output',
            help='The output directory.',
            type=str,
            required=True)

    args = parser.parse_args()

    return args


def read_annotations(filename, num_points=10):
    """Read the annotations from file.
    """
    annotations = []

    with open(filename, 'r') as fin:
        for line in fin:
            items = line.split(' ')

            points = []
            for i in xrange(num_points):
                x = float(items[4 + 2*i])
                y = float(items[4 + 2*i + 1])
                points.append([x, y])

            entry = {
                    'video': items[0],
                    'image': items[1],
                    'image_shape': (float(items[3]), float(items[2])),
                    'points': points
                    }
            annotations.append(entry)

    return annotations


def convert_annotations(annotations):
    """convert the annotations.
    """
    new_annotations = []

    for entry in annotations:
        new_entry = {}

        items = entry['video'].split('_')
        assert len(items) == 3
        sub_items = items[0].split('-')

        # Create the new entry.
        new_entry['channel'] = sub_items[0]
        new_entry['playlist'] = sub_items[1]
        new_entry['video'] = sub_items[2]
        new_entry['start_time'] = float(items[1])
        new_entry['end_time'] = float(items[2][:-4])
        new_entry['image'] = entry['image'].replace(items[0], '')[2:]
        new_entry['image_shape'] = entry['image_shape']
        new_entry['points'] = entry['points']

        new_annotations.append(new_entry)

    return new_annotations


def write_annotations(filename, annotations):
    """Read the annotations from file.
    """
    with open(filename, 'w') as fout:
        for entry in annotations:
            line = '%s %s %s %s %.1f %.1f' % (
                    entry['channel'], entry['playlist'], entry['video'],
                    entry['image'],
                    entry['image_shape'][0], entry['image_shape'][1])

            for point in entry['points']:
                line += ' %.1f %.1f' % (point[0], point[1])

            line += '\n'

            fout.write(line)


def main():
    args = parse_args()

    annotations = read_annotations(args.input, num_points=10)
    new_annotations = convert_annotations(annotations)
    write_annotations(args.output, new_annotations)


if __name__ == '__main__':
    main()
