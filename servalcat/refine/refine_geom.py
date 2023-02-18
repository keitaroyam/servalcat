"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import argparse
import os
import gemmi
import numpy
import servalcat # for version
from servalcat.utils import logger
from servalcat import utils
from servalcat.refine.refine import Geom, Refine

def add_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model',
                        help='Input atomic model file')
    group.add_argument('--update_dictionary', 
                       help="Dictionary file to be updated")
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
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('-o','--output_prefix')

# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def main(args):
    if args.model:
        if not args.output_prefix:
            args.output_prefix = utils.fileio.splitext(os.path.basename(args.model))[0] + "_refined"
        st = utils.fileio.read_structure(args.model)
        utils.model.setup_entities(st, clear=True, force_subchain_names=True)
        st.assign_cis_flags()
        if st.ncs:
            st2 = st.clone()
            logger.writeln("Take NCS constraints into account.")
            st2.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
            utils.fileio.write_model(st2, file_name="input_expanded.pdb")

        if args.ligand: args.ligand = sum(args.ligand, [])
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand,
                                                       stop_for_unknowns=True)
        utils.restraints.find_and_fix_links(st, monlib) # should remove unknown id here?
        h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                    "full":gemmi.HydrogenChange.ReAdd,
                    "yes":gemmi.HydrogenChange.NoChange,
                    "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
    else:
        if not args.output_prefix:
            args.output_prefix = utils.fileio.splitext(os.path.basename(args.update_dictionary))[0] + "_refined"
        doc = gemmi.cif.read(args.update_dictionary)
        for block in doc: # this block will be reused below
            st = gemmi.make_structure_from_chemcomp_block(block)
            if len(st) > 0: break
        else:
            raise SystemExit("No model in the cif file")
        if args.ligand:
            logger.writeln("WARNING: monlib and ligand are ignored in the dictionary updating mode")
            args.ligand = [args.update_dictionary]
        monlib = utils.restraints.load_monomer_library(st, monomer_dir=args.monlib, # monlib is needed for ener_lib
                                                       cif_files=[args.update_dictionary],
                                                       stop_for_unknowns=True)
        h_change = gemmi.HydrogenChange.NoChange
        
    try:
        topo = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                 check_hydrogen=(args.hydrogen=="yes"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if args.hydrogen == "full":
        for cra in st[0].all():
            if cra.atom.is_hydrogen(): cra.atom.occ = 1.

    keywords = []
    if args.keywords or args.keyword_file:
        if args.keywords: keywords = sum(args.keywords, [])
        if args.keyword_file: keywords.extend(l for f in sum(args.keyword_file, []) for l in open(f))

    geom = Geom(st, topo, monlib, shake_rms=args.randomize, refmac_keywords=keywords)
    refiner = Refine(st, geom)
    refiner.run_cycles(args.ncycle)
    utils.fileio.write_model(refiner.st, args.output_prefix, pdb=True, cif=True)

    if args.update_dictionary:
        # replace xyz
        pos = {cra.atom.name: cra.atom.pos.tolist() for cra in refiner.st[0].all()}
        for row in block.find("_chem_comp_atom.", ["atom_id", "x", "y", "z"]):
            p = pos[row.str(0)]
            for i in range(3):
                row[i+1] = "{:.3f}".format(p[i])
        # add description
        loop = block.find_loop("_pdbx_chem_comp_description_generator.comp_id").get_loop()
        if not loop:
            loop = block.init_loop("_pdbx_chem_comp_description_generator.", ["comp_id",
                                                                              "program_name",
                                                                              "program_version",
                                                                              "descriptor"])
        tags = [x[x.index(".")+1:] for x in loop.tags]
        row = ["" for _ in range(len(tags))]
        for tag, val in (("comp_id", st[0][0][0].name),
                         ("program_name", "servalcat"),
                         ("program_version", servalcat.__version__),
                         ("descriptor", "optimization tool")):
            if tag in tags: row[tags.index(tag)] = val
        loop.add_row(gemmi.cif.quote_list(row))
        doc.write_file(args.output_prefix + "_updated.cif", style=gemmi.cif.Style.Aligned)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
