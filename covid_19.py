#! /usr/bin/env python

import sys
import argparse
from models.prep_data import *
from models.linear import *
from models.model_cases import *
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
    parser.add_argument('-m', '--model', nargs='?', const='linear', type = str,
                        help='Choose statistical model. Default is linear.')
    parser.add_argument('-r', '--region', nargs='?', default='all',
                        type = str, help='Choose country, country code, geoid, continent or all. Default is all.')
    parser.add_argument('-t', '--target', nargs='?', default='deaths', choices = ['deaths', 'cases'],
                        type = str, help='Choose the target. Default is deaths.')
    parser.add_argument('-o', '--outliers_method', nargs='?', default='knn',
                        choices = ['knn', 'LocalOutlierFactor', 'EllipticEnvelope'],
                        type = str, help='Choose method for the detection of outliers. Default is knn.')
    parser.add_argument('-op', '--outliers_portion', nargs='?', default= 0.1,
                        type = float, help='Provide the portion of outliers in the dataset. Default is 0.1.')
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
    if args.target == 'cases':
        data = dataset(args.region)
        model_cases(data)
    elif args.model in ELIG_MODELS:
        data = dataset(args.region)
        clean_data = outlier_detection(data, args.outliers_portion, args.outliers_method)
        linear(clean_data)
    else:
        print(f'The model provided is not supported. please see README.md for'
               ' supported statistical models. Program exits.')
        sys.exit(-1)


if __name__ == '__main__':
    detect_python_version()
    main()
