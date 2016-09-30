import argparse
import os

import suspect


def anonymize_twix():

    # start with a simple parser which looks for the single positional argument, the path to the graph file
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("output_filename")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print("anonymize_twix: cannot anonymize '{0}': No such file".format(args.filename))
        exit(-1)

    try:
        suspect.io.twix.anonymize_twix(args.filename, args.output_filename)
    except Exception as e:
        print("anonymize_twix: cannot anonymize '{0}': {1}".format(args.filename, e))
