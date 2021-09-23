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
import gemmi
import numpy
import scipy
import pandas
import servalcat.spa.sfcalc
import servalcat.spa.shiftback
import servalcat.spa.run_refmac
import servalcat.spa.fsc
import servalcat.spa.fofc
import servalcat.spa.shift_maps
import servalcat.spa.translate
import servalcat.utils.commands
#import servalcat.spa.bestmap

from servalcat.utils import logger

def dependency_versions():
    return dict(gemmi=gemmi.__version__,
                scipy=scipy.version.full_version,
                numpy=numpy.version.full_version,
                pandas=pandas.__version__)
# dependency_versions()

def main():
    
    parser = argparse.ArgumentParser(prog="servalcat",
                                     description="A tool for model refinement and map calculation for cryo-EM SPA.")
    parser.add_argument("-v", "--version", action="version",
                        version="Servalcat {servalcat} with Python {python} ({deps})".format(servalcat=servalcat.__version__,
                                                                                             python=platform.python_version(),
                                                                                             deps=", ".join([x[0]+" "+x[1] for x in dependency_versions().items()])))
    subparsers = parser.add_subparsers(dest="command")

    modules = dict(sfcalc=servalcat.spa.sfcalc,
                   shiftback=servalcat.spa.shiftback,
                   refine_spa=servalcat.spa.run_refmac,
                   fsc=servalcat.spa.fsc,
                   fofc=servalcat.spa.fofc,
                   trim=servalcat.spa.shift_maps,
                   translate=servalcat.spa.translate,
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
        logger.write("# Servalcat ver. {} (Python {})".format(servalcat.__version__, platform.python_version()))
        logger.write("# Library vers. {}".format(", ".join([x[0]+" "+x[1] for x in dependency_versions().items()])))
        logger.write("# Started on {}".format(datetime.datetime.now()))
        logger.write("# Host: {} User: {}".format(platform.node(), getpass.getuser()))
        logger.write("# Command-line args:")
        logger.write("# {}".format(" ".join(map(lambda x: pipes.quote(x), sys.argv[1:]))))
        modules[args.command].main(args)
        logger.write("\n# Finished on {}\n".format(datetime.datetime.now()))
    else:
        parser.print_help()

# main()

if __name__ == "__main__":
    main()

