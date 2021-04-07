"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import fileio
from servalcat.utils import symmetry
from servalcat.utils import restraints
import os
import gemmi

def add_arguments(p):
    subparsers = p.add_subparsers(dest="subcommand")
    
    parser = subparsers.add_parser("show", description = 'Show file info supported by the program')
    parser.add_argument('files', nargs='+')

    parser = subparsers.add_parser("symmodel", description="Add symmetry annotation to model")
    parser.add_argument('--model', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--map', help="Take box size from the map")
    group.add_argument('--cell', type=float, nargs=6, help="Box size")
    parser.add_argument('--pg', required=True, help="Point group symbol")
    parser.add_argument('--chains', nargs="*", action="append", help="Select chains to keep")
    parser.add_argument('--biomt', action="store_true", help="Add BIOMT also")
    parser.add_argument('-o', '--output_prfix')
    parser.add_argument('--pdb', action="store_true", help="Write a pdb file")
    parser.add_argument('--cif', action="store_true", help="Write a cif file")
    
    parser = subparsers.add_parser("h_add", description = 'Add hydrogen in riding position')
    parser.add_argument('model')
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('-o','--output')
    parser.add_argument("--pos", choices=["elec", "nucl"], default="elec")

    parser = subparsers.add_parser("merge_models", description = 'Merge multiple model files')
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o','--output', required=True)
# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def symmodel(args):
    if args.chains: args.chains = sum(args.chains, [])
    model_format = fileio.check_model_format(args.model)
    _, _, ops = symmetry.operators_from_symbol(args.pg)
    logger.write("{} operators found for {}".format(len(ops), args.pg))
    symmetry.show_operators_axis_angle(ops)
    st = gemmi.read_structure(args.model)
    st.spacegroup_hm = "P 1"
    if args.map:
        logger.write("Reading cell from map")
        g, _ = fileio.read_ccp4_map(args.map)
        st.cell = g.unit_cell
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        logger.write("Error: Unit cell parameters look wrong. Please use --map or --cell")
        return

    if args.chains:
        logger.write("Keep {} chains only".format(" ".join(args.chains)))
        chains = set(args.chains)
        for m in st:
            to_del = [c.name for c in m if c.name not in chains]
            for c in to_del: m.remove_chain(c)

    all_chains = [c.name for c in st[0] if c.name not in st[0]]
    ncsops = symmetry.make_NcsOps_from_matrices(ops, cell=st.cell)
    st.ncs.clear()
    st.ncs.extend([x for x in ncsops if not x.tr.is_identity()])

    if args.biomt:
        st.assemblies.clear()
        st.raw_remarks = []
        a = gemmi.Assembly("1")
        g = gemmi.Assembly.Gen()
        for i, nop in enumerate(ncsops):
            op = gemmi.Assembly.Operator()
            op.transform = nop.tr
            if not nop.tr.is_identity(): op.type = "point symmetry operation"
            g.operators.append(op)
        g.chains = all_chains
        a.generators.append(g)
        a.special_kind = gemmi.AssemblySpecialKind.CompletePoint
        st.assemblies.append(a)

    if not args.output_prfix:
        args.output_prfix = fileio.splitext(os.path.basename(args.model))[0] + "_asu"

    if args.pdb or args.cif:
        fileio.write_model(st, args.output_prfix, pdb=args.pdb, cif=args.cif)
    else:
        fileio.write_model(st, file_name=args.output_prfix+model_format)

    # Sym expand
    st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
    st.assemblies.clear()
    args.output_prfix += "_expanded"
    if args.pdb or args.cif:
        fileio.write_model(st, args.output_prfix, pdb=args.pdb, cif=args.cif)
    else:
        fileio.write_model(st, file_name=args.output_prfix+model_format)
# symmodel()

def h_add(args):
    st = gemmi.read_structure(args.model)
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
    restraints.add_hydrogens(st, monlib, args.pos)
    fileio.write_model(st, file_name=args.output)
# h_add()

def merge_models(args):
    #sts = [gemmi.read_structure(f) for f in args.models]
    logger.write("Reading file   1: {}".format(args.models[0]))
    st = gemmi.read_structure(args.models[0])
    logger.write("                  chains {}".format(" ".join([c.name for c in st[0]])))

    for i, f in enumerate(args.models[1:]):
        logger.write("Reading file {:3d}: {}".format(i+2, f))
        st2 = gemmi.read_structure(f)
        for c in st2[0]:
            org_id = c.name
            c2 = st[0].add_chain(c, unique_name=True)
            if c.name != c2.name:
                logger.write("                  chain {} merged (ID changed to {})".format(c.name, c2.name))
            else:
                logger.write("                  chain {} merged".format(c.name))

    fileio.write_model(st, file_name=args.output)
# merge_models()

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
    elif com == "symmodel":
        symmodel(args)
    elif com == "h_add":
        h_add(args)
    elif com == "merge_models":
        merge_models(args)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
