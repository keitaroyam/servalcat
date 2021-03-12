"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import fileio
from servalcat.utils import restraints
import os
import gemmi

def add_arguments(p):
    subparsers = p.add_subparsers(dest="subcommand")
    parser = subparsers.add_parser("show", description = 'Show file info supported by the program')
    parser.add_argument('files', nargs='+')

    parser = subparsers.add_parser("h_add", description = 'Add hydrogen in riding position')
    parser.add_argument('model')
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('-o','--output')
    parser.add_argument("--pos", choices=["elec", "nucl"], default="elec")


# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def h_add(args):
    st = gemmi.read_structure(args.model)
    st.setup_entities()
    resnames = st[0].get_all_residue_names()
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_h" + model_format
        logger.write("Output file: {}".format(args.output))
        
        
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(resnames,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)

    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAddButWater)
    if args.pos == "nucl":
        logger.write("Adjusting distance to nucleus")
        restraints.check_monlib_support_nucleus_distances(monlib, resnames)
        topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus)

    fileio.write_model(st, file_name=args.output)
# h_add()

def show(args):
    for filename in args.files:
        ext = fileio.splitext(filename)[1]
        if ext in (".mrc", ".ccp4", ".map"):
            fileio.read_ccp4_map(filename)
            logger.write("\n")
# show()

def main(args):
    com = args.subcommand
    if com == "show":
        show(args)
    elif com == "h_add":
        h_add(args)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
