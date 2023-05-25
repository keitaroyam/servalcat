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
import json
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
    parser.add_argument('--hydrogen', default="all", choices=["all", "yes", "no"],
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

def refine_and_update_dictionary(cif_in, monomer_dir, output_prefix, randomize=0, ncycle1=10, ncycle2=30):
    doc = gemmi.cif.read(cif_in)
    for block in doc: # this block will be reused below
        st = gemmi.make_structure_from_chemcomp_block(block)
        if len(st) > 0: break
    else:
        raise SystemExit("No model in the cif file")
    monlib = utils.restraints.load_monomer_library(st, monomer_dir=monomer_dir, # monlib is needed for ener_lib
                                                   cif_files=[cif_in],
                                                   stop_for_unknowns=True)
    try:
        topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAdd,
                                                    check_hydrogen=False)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    geom = Geom(st, topo, monlib, shake_rms=randomize)
    refiner = Refine(st, geom)
    logger.writeln("Running {} cycles with wchir=4 wvdw=2".format(ncycle1))
    geom.calc_kwds["wchir"] = 4
    geom.calc_kwds["wvdw"] = 2
    refiner.run_cycles(ncycle1)

    # re-add hydrogen may help
    topo = gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAdd,
                                  warnings=logger)
    geom = Geom(st, topo, monlib)
    refiner = Refine(st, geom)
    logger.writeln("Running {} cycles with wchir=1 wvdw=2".format(ncycle2))
    geom.calc_kwds["wchir"] = 1
    geom.calc_kwds["wvdw"] = 2
    refiner.run_cycles(ncycle2)

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
    doc.write_file(output_prefix + "_updated.cif", style=gemmi.cif.Style.Aligned)
# refine_and_update_dictionary()

def refine_geom(model_in, monomer_dir, cif_files, h_change, ncycle, output_prefix, randomize, refmac_keywords):
    st = utils.fileio.read_structure(model_in)
    utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    if st.ncs:
        st2 = st.clone()
        logger.writeln("Take NCS constraints into account.")
        st2.expand_ncs(gemmi.HowToNameCopiedChain.Dup)
        utils.fileio.write_model(st2, file_name="input_expanded.pdb")

    monlib = utils.restraints.load_monomer_library(st, monomer_dir=monomer_dir,
                                                   cif_files=cif_files,
                                                   stop_for_unknowns=True)
    utils.restraints.find_and_fix_links(st, monlib) # should remove unknown id here?
    try:
        topo, metal_kws = utils.restraints.prepare_topology(st, monlib, h_change=h_change,
                                                            check_hydrogen=(h_change==gemmi.HydrogenChange.NoChange))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    refmac_keywords.extend(metal_kws)
    geom = Geom(st, topo, monlib, shake_rms=randomize, refmac_keywords=refmac_keywords)
    refiner = Refine(st, geom)
    stats = refiner.run_cycles(ncycle)
    refiner.st.name = output_prefix
    utils.fileio.write_model(refiner.st, output_prefix, pdb=True, cif=True)
    with open(output_prefix + "_stats.json", "w") as ofs:
        for s in stats: s["geom"] = s["geom"].to_dict()
        json.dump(stats, ofs, indent=2)
        logger.writeln("Refinement statistics saved: {}".format(ofs.name))
# refine_geom()

def main(args):
    keywords = []
    if args.keywords: keywords = sum(args.keywords, [])
    if args.keyword_file: keywords.extend(l for f in sum(args.keyword_file, []) for l in open(f))
    decide_prefix = lambda f: utils.fileio.splitext(os.path.basename(f))[0] + "_refined"
    if args.model:
        if not args.output_prefix:
            args.output_prefix = decide_prefix(args.model)
        if args.ligand:
            args.ligand = sum(args.ligand, [])
        h_change = {"all":gemmi.HydrogenChange.ReAddButWater,
                    "full":gemmi.HydrogenChange.ReAdd,
                    "yes":gemmi.HydrogenChange.NoChange,
                    "no":gemmi.HydrogenChange.Remove}[args.hydrogen]
        refine_geom(model_in=args.model,
                    monomer_dir=args.monlib,
                    cif_files=args.ligand,
                    h_change=h_change,
                    ncycle=args.ncycle,
                    output_prefix=args.output_prefix,
                    randomize=args.randomize,
                    refmac_keywords=keywords)
    else:
        if not args.output_prefix:
            args.output_prefix = decide_prefix(args.update_dictionary)
        if args.ligand:
            logger.writeln("WARNING: monlib and ligand are ignored in the dictionary updating mode")
        if keywords:
            logger.writeln("WARNING: refmac keywords are ignored in the dictionary updating mode")
        refine_and_update_dictionary(cif_in=args.update_dictionary,
                                     monomer_dir=args.monlib,
                                     output_prefix=args.output_prefix,
                                     randomize=args.randomize)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
