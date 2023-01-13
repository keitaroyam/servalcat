"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import gemmi
import numpy
import scipy.optimize
import scipy.sparse
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine.refine import Refine

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

def add_arguments(parser):
    parser.add_argument('--model', required=True,
                        help='Input atomic model file')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append",
                        help="restraint dictionary cif file(s)")
    parser.add_argument('--ncycle', type=int, default=10,
                        help="number of CG cycles (default: %(default)d)")
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no", "full"],
                        help="all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. "
                        "Default: %(default)s")
    parser.add_argument('--randomize', type=float, default=0,
                        help='Shake coordinates with specified rmsd')

# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    st = utils.fileio.read_structure(args.model)
    logger.writeln("NCS= {}".format([x for x in st.ncs]))
    if st.ncs:
        st2 = st.clone()
        logger.writeln("Take NCS constraints into account.")
        st2.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        utils.fileio.write_model(st2, file_name="input_expanded.pdb")

    if args.ligand: args.ligand = sum(args.ligand, [])
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                   stop_for_unknowns=True,
                                                   check_hydrogen=(args.hydrogen=="yes"))
    h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                "full":gemmi.HydrogenChange.ReAdd,
                "yes":gemmi.HydrogenChange.NoChange,
                "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger,
                                  reorder=True, ignore_unknown_links=False) # we should remove logger here??
    if args.hydrogen == "full":
        for cra in st[0].all():
            if cra.atom.is_hydrogen(): cra.atom.occ = 1.

    refiner = Refine(st, topo, monlib)

    if args.randomize > 0:
        numpy.random.seed(0)
        from servalcat.utils import model
        utils.model.shake_structure(refiner.st, args.randomize, copy=False)

    for i in range(args.ncycle):
        logger.writeln("==== CYCLE {:2d}".format(i))
        refiner.run_cycle()
        utils.fileio.write_model(refiner.st, "refined_{:02d}".format(i), pdb=True)#, cif=True)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
