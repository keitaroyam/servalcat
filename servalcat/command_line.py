"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import sys
import datetime
import pipes
import servalcat.spa.sfcalc
import servalcat.spa.shiftback
import servalcat.spa.run_refmac
import servalcat.spa.fsc
import servalcat.spa.fofc
import servalcat.spa.shift_maps
#import servalcat.spa.bestmap

from servalcat.utils import logger

def main():
    logger.set_file("servalcat.log")
    logger.write("# Started on {}".format(datetime.datetime.now()))
    logger.write("# Command-line args:")
    logger.write("# {}".format(" ".join(map(lambda x: pipes.quote(x), sys.argv[1:]))))

    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", description='')

    modules = dict(sfcalc=servalcat.spa.sfcalc,
                   shiftback=servalcat.spa.shiftback,
                   refine_spa=servalcat.spa.run_refmac,
                   fsc=servalcat.spa.fsc,
                   fofc=servalcat.spa.fofc,
                   shift=servalcat.spa.shift_maps,
                   show=servalcat.utils.show,
                   #bestmap=servalcat.spa.bestmap,
                   )

    for n in modules:
        p = subparsers.add_parser(n)
        modules[n].add_arguments(p)

    args = parser.parse_args()
    
    if args.command in modules:
        modules[args.command].main(args)
    else:
        parser.print_help()

    logger.write("# Finished on {}\n".format(datetime.datetime.now()))
# main()

if __name__ == "__main__":
    main()

