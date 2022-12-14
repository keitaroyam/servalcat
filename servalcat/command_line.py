"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import sys
import traceback
import platform
import gemmi
import numpy
import scipy
import pandas
import servalcat.spa.shiftback
import servalcat.spa.run_refmac
import servalcat.spa.fsc
import servalcat.spa.fofc
import servalcat.spa.shift_maps
import servalcat.spa.translate
import servalcat.spa.localcc
import servalcat.xtal.run_refmac_small
import servalcat.xtal.sigmaa
import servalcat.refmac.refmac_wrapper
import servalcat.utils.commands

from servalcat.utils import logger

def test_installation():
    vers = logger.dependency_versions()
    pandas_ver = [int(x) for x in vers["pandas"].split(".")]
    numpy_ver = [int(x) for x in vers["numpy"].split(".")]
    msg_unknown = "Unexpected error occurred (related to numpy+pandas). Please report to authors with the result of servalcat -v."
    msg_skip = "If you want to ignore this error, please specify --skip_test."
    ret = True
    
    try:
        x = pandas.DataFrame(dict(x=[2j]))
        x.merge(x)
    except TypeError:
        ret = False
        if pandas_ver >= [1,3,0] and numpy_ver < [1,19,1]:
            print("There is a problem in pandas+numpy. Please update numpy to 1.19.1 or newer (or use pandas < 1.3.0).")
        else:
            print(traceback.format_exc())
            print(msg_unknown)
    except:
        print(traceback.format_exc())
        print(msg_unknown)
        ret = False

    if not ret:
        print(msg_skip)
        
    return ret
# test_installation()        

def main():
    parser = argparse.ArgumentParser(prog="servalcat",
                                     description="A tool for model refinement and map calculation for cryo-EM SPA.")
    parser.add_argument("--skip_test", action="store_true", help="Skip installation test")
    parser.add_argument("-v", "--version", action="version",
                        version="Servalcat {servalcat} with Python {python} ({deps})".format(servalcat=servalcat.__version__,
                                                                                             python=platform.python_version(),
                                                                                             deps=", ".join([x[0]+" "+x[1] for x in logger.dependency_versions().items()])))
    subparsers = parser.add_subparsers(dest="command")

    modules = dict(shiftback=servalcat.spa.shiftback,
                   refine_spa=servalcat.spa.run_refmac,
                   refine_cx=servalcat.xtal.run_refmac_small,
                   fsc=servalcat.spa.fsc,
                   fofc=servalcat.spa.fofc,
                   trim=servalcat.spa.shift_maps,
                   translate=servalcat.spa.translate,
                   localcc=servalcat.spa.localcc,
                   sigmaa=servalcat.xtal.sigmaa,
                   #show=servalcat.utils.show,
                   util=servalcat.utils.commands,
                   refmac5=servalcat.refmac.refmac_wrapper,
                   )

    for n in modules:
        p = subparsers.add_parser(n)
        modules[n].add_arguments(p)

    args = parser.parse_args()
    
    if not args.skip_test and not test_installation():
        return
    
    if args.command == "util" and not args.subcommand:
        print("specify subcommand.")    
    elif args.command in modules:
        logger.set_file("servalcat.log")
        logger.write_header()
        try:
            modules[args.command].main(args)
        except SystemExit as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        parser.print_help()

# main()

if __name__ == "__main__":
    main()

