"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat import utils
from servalcat.utils import logger

def add_arguments(parser):
    parser.description = 'Show file info supported by the program'
    parser.add_argument('files', nargs='*')
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def show(filename):
    ext = utils.fileio.splitext(filename)[1]
    if ext in (".mrc", ".ccp4", ".map"):
        utils.fileio.read_ccp4_map(filename)
    logger.write("\n")
# show()

def main(args):
    for f in args.files:
        show(f)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
