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

    sfcalc = subparsers.add_parser('sfcalc')
    servalcat.spa.sfcalc.add_arguments(sfcalc)
    shiftback = subparsers.add_parser('shiftback')
    servalcat.spa.shiftback.add_arguments(shiftback)
    refine_spa = subparsers.add_parser('refine_spa')
    servalcat.spa.run_refmac.add_arguments(refine_spa)
    fsc = subparsers.add_parser('fsc')
    servalcat.spa.fsc.add_arguments(fsc)
    fofc = subparsers.add_parser('fofc')
    servalcat.spa.fofc.add_arguments(fofc)
    shift_maps = subparsers.add_parser('shift')
    servalcat.spa.shift_maps.add_arguments(shift_maps)
    #bestmap = subparsers.add_parser('bestmap')
    #servalcat.spa.bestmap.add_arguments(bestmap)

    args = parser.parse_args()

    if args.command == "sfcalc":
        servalcat.spa.sfcalc.main(args)
    elif args.command == "shiftback":
        servalcat.spa.shiftback.main(args)
    elif args.command == "refine_spa":
        servalcat.spa.run_refmac.main(args)
    elif args.command == "fsc":
        servalcat.spa.fsc.main(args)
    elif args.command == "fofc":
        servalcat.spa.fofc.main(args)
    elif args.command == "shift":
        servalcat.spa.shift_maps.main(args)
    #elif args.command == "bestmap":
    #    servalcat.spa.bestmap.main(args)
    else:
        parser.print_help()
        
    logger.write("# Finished on {}".format(datetime.datetime.now()))
# main()

if __name__ == "__main__":
    main()

