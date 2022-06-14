#!/usr/bin/env python3

"""
"""
import os

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd

def main(args):
    training_record_dir = os.path.join(args.training_directory, 'training.csv')
    training_record = pd.read_csv(training_record_dir)
    epoch = training_record.sort_values(["validation_median","validation_mean"]).iloc[-1, 2]
    print(epoch)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    return parser


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args)
