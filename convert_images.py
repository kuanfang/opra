#!/usr/bin/env python

"""Convert images from old directory to new directory.

This is used to convert the original annotations for data releasing.

Author: Te-Lin Wu, Kuan Fang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import shutil


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input',
            dest='input_dir',
            help='The input directory',
            type=str,
            required=True)

    parser.add_argument(
            '--output',
            dest='output_dir',
            help='The output directory.',
            type=str,
            required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    for input_path in glob.glob(os.path.join(args.input_dir, '*', '*')):
        items = input_path.split('/')
        filename = items[2].replace(items[1], '')[1:]
        sub_items = items[1].split('-')

        output_dir = os.path.join(
                args.output_dir, sub_items[0], sub_items[1], sub_items[2])
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Copying image from %s to %s...' % (input_path, output_dir))
        shutil.copy(input_path, output_path)


if __name__ == '__main__':
    main()

