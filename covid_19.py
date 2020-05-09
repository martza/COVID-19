#! /usr/bin/env python

import sys
import argparse
from models import linear


# constants
ELIG_MODELS = ['linear', 'non-linear']


def detect_python_version():
    """
    """
    if sys.version_info.major < 3:
        print(f'Python {sys.version_info.major} '
                f'is not supported. Program exits.')
        sys.exit(-1)


def parse_arguments():
    """
    parse_arguments:
    Args: -
    Returns: args [str]
    """
    parser = argparse.ArgumentParser(
                description='Parse various arguments.'
                )
    parser.add_argument('-m', '--model', nargs='?', const='linear',
                help='choose statistical model.')
    args = parser.parse_args()
    return(args)  


def main(args=None):
    """
    main:
    Args: args [str]
    Returns: -
    This is the main function. It may receive a list of strings as
    input if it is called by another function. By default the arguments
    are set to None so that the parser is called in case that the user is
    executing this main script.
    """

    if args == None:
        args = parse_arguments()

    # input
    if args.model in ELIG_MODELS:
        linear()
    else:
        print(f'The model provided is not supported. please see README.md for'
               ' supported statistical models. Program exits.')
        sys.exit(-1)


if __name__ == '__main__':
    detect_python_version()
    main()