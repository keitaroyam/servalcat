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
import getpass
import platform
import servalcat.spa.sfcalc
import servalcat.spa.shiftback
import servalcat.spa.run_refmac
import servalcat.spa.fsc
import servalcat.spa.fofc
import servalcat.spa.shift_maps
import servalcat.utils.commands
#import servalcat.spa.bestmap

from servalcat.utils import logger

def main():
    
    parser = argparse.ArgumentParser(description="")
#Some useful commands:
#  trim        Trim maps and shift models to reduce file size
#  refina_spa  Refine CryoEM SPA structure using REFMAC5
#  fofc        Calculate updated map and Fo-Fc map using error estimates from half maps and model

    subparsers = parser.add_subparsers(dest="command")

    modules = dict(sfcalc=servalcat.spa.sfcalc,
                   shiftback=servalcat.spa.shiftback,
                   refine_spa=servalcat.spa.run_refmac,
                   fsc=servalcat.spa.fsc,
                   fofc=servalcat.spa.fofc,
                   trim=servalcat.spa.shift_maps,
                   #show=servalcat.utils.show,
                   util=servalcat.utils.commands,
                   #bestmap=servalcat.spa.bestmap,
                   )

    for n in modules:
        p = subparsers.add_parser(n)
        modules[n].add_arguments(p)

    args = parser.parse_args()
    
    if args.command == "util" and not args.subcommand:
        print("specify subcommand.")    
    elif args.command in modules:
        logger.set_file("servalcat.log")
        logger.write("# Servalcat ver. {}".format(servalcat.__version__))
        logger.write("# Started on {}".format(datetime.datetime.now()))
        logger.write("# Host: {} User: {}".format(platform.node(), getpass.getuser()))
        logger.write("# Command-line args:")
        logger.write("# {}".format(" ".join(map(lambda x: pipes.quote(x), sys.argv[1:]))))
        modules[args.command].main(args)
        logger.write("# Finished on {}\n".format(datetime.datetime.now()))
    else:
        parser.print_help()

# main()

if __name__ == "__main__":
    main()

