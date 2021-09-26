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
from servalcat.utils import model
from servalcat.utils import restraints
from servalcat.utils import maps
import os
import gemmi
import numpy
import pandas

def add_arguments(p):
    subparsers = p.add_subparsers(dest="subcommand")

    # show
    parser = subparsers.add_parser("show", description = 'Show file info supported by the program')
    parser.add_argument('files', nargs='+')

    # json2csv
    parser = subparsers.add_parser("json2csv", description = 'Convert json to csv for plotting')
    parser.add_argument('json')
    parser.add_argument('-o', '--output_prefix')

    # symmodel
    parser = subparsers.add_parser("symmodel", description="Add symmetry annotation to model")
    parser.add_argument('--model', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--map', help="Take box size from the map")
    group.add_argument('--cell', type=float, nargs=6, help="Box size")
    parser.add_argument('--pg', required=True, help="Point group symbol")
    parser.add_argument('--twist', type=float, help="Helical twist (degree)")
    parser.add_argument('--rise', type=float, help="Helical rise (Angstrom)")
    parser.add_argument('--center', type=float, nargs=3, help="Origin of symmetry. Default: center of the box")
    parser.add_argument('--axis1', type=float, nargs=3, help="Axis1 (if I: 5-fold, O: 4-fold, T: 3-fold)")
    parser.add_argument('--axis2', type=float, nargs=3, help="Axis2 (if I: 5-fold, O: 4-fold, T: 3-fold)")
    parser.add_argument('--chains', nargs="*", action="append", help="Select chains to keep")
    parser.add_argument('--howtoname', choices=["dup", "short", "number"], default="short",
                        help="How to decide new chain IDs in expanded model (default: short); "
                        "dup: use original chain IDs (with different segment IDs), "
                        "short: use unique new IDs, "
                        "number: add number to original chain ID")
    parser.add_argument('--biomt', action="store_true", help="Add BIOMT also")
    parser.add_argument('-o', '--output_prfix')
    parser.add_argument('--pdb', action="store_true", help="Write a pdb file")
    parser.add_argument('--cif', action="store_true", help="Write a cif file")

    # expand
    parser = subparsers.add_parser("expand", description="Expand symmetry")
    parser.add_argument('--model', required=True)
    parser.add_argument('--chains', nargs="*", action="append", help="Select chains to keep")
    parser.add_argument('--howtoname', choices=["dup", "short", "number"], default="short",
                        help="How to decide new chain IDs in expanded model (default: short); "
                        "dup: use original chain IDs (with different segment IDs), "
                        "short: use unique new IDs, "
                        "number: add number to original chain ID")
    parser.add_argument('-o', '--output_prfix')
    parser.add_argument('--pdb', action="store_true", help="Write a pdb file")
    parser.add_argument('--cif', action="store_true", help="Write a cif file")

    # h_add
    parser = subparsers.add_parser("h_add", description = 'Add hydrogen in riding position')
    parser.add_argument('model')
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('-o','--output')
    parser.add_argument("--pos", choices=["elec", "nucl"], default="elec")

    # merge_models
    parser = subparsers.add_parser("merge_models", description = 'Merge multiple model files')
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o','--output', required=True)

    # power
    parser = subparsers.add_parser("power", description = 'Show power spectrum')
    parser.add_argument("--map",  nargs="*", action="append")
    parser.add_argument("--halfmaps",  nargs="*", action="append")
    parser.add_argument('--mask', help='Mask file')
    parser.add_argument('-d', '--resolution', type=float, required=True)
    parser.add_argument('-o', '--output_prefix', default="power")

    # fcalc
    parser = subparsers.add_parser("fcalc", description = 'Structure factor from model')
    parser.add_argument('--model', required=True)
    parser.add_argument("--source", choices=["electron", "xray"], default="electron")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--cell', type=float, nargs=6, help="Override unit cell")
    parser.add_argument('--cutoff', type=float, default=1e-7)
    parser.add_argument('--rate', type=float, default=1.5)
    parser.add_argument('-d', '--resolution', type=float, required=True)
    parser.add_argument('-o', '--output_prefix')

# add_arguments()

def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def symmodel(args):
    if args.chains: args.chains = sum(args.chains, [])
    model_format = fileio.check_model_format(args.model)

    howtoname = dict(dup=gemmi.HowToNameCopiedChain.Dup,
                     short=gemmi.HowToNameCopiedChain.Short,
                     number=gemmi.HowToNameCopiedChain.AddNumber)[args.howtoname]

    if (args.twist, args.rise).count(None) == 1:
        logger.error("ERROR: give both helical paramters --twist and --rise")
        return

    is_helical = args.twist is not None

    st, cif_ref = fileio.read_structure_from_pdb_and_mmcif(args.model)
    st.spacegroup_hm = "P 1"
    start_xyz = numpy.zeros(3)
    if args.map:
        logger.write("Reading cell from map")
        g, grid_start = fileio.read_ccp4_map(args.map)
        st.cell = g.unit_cell
        start_xyz = numpy.array(g.get_position(*grid_start).tolist())
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        logger.error("Error: Unit cell parameters look wrong. Please use --map or --cell")
        return

    if args.chains:
        logger.write("Keep {} chains only".format(" ".join(args.chains)))
        chains = set(args.chains)
        for m in st:
            to_del = [c.name for c in m if c.name not in chains]
            for c in to_del: m.remove_chain(c)

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    if args.center is None:
        A = numpy.array(st.cell.orthogonalization_matrix.tolist())
        center = numpy.sum(A, axis=1) / 2 + start_xyz
        logger.write("Center: {}".format(center))
    else:
        center = numpy.array(args.center)

    if is_helical:
        ncsops = symmetry.generate_helical_operators(st, start_xyz, center,
                                                     args.pg, args.twist, args.rise,
                                                     axis1=args.axis1, axis2=args.axis2)
        logger.write("{} helical operators found".format(len(ncsops)))
    else:
        _, _, ops = symmetry.operators_from_symbol(args.pg, axis1=args.axis1, axis2=args.axis2)
        logger.write("{} operators found for {}".format(len(ops), args.pg))
        symmetry.show_operators_axis_angle(ops)
        ncsops = symmetry.make_NcsOps_from_matrices(ops, cell=st.cell, center=center)

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
        fileio.write_model(st, args.output_prfix, pdb=args.pdb, cif=args.cif, cif_ref=cif_ref)
    else:
        fileio.write_model(st, file_name=args.output_prfix+model_format, cif_ref=cif_ref)

    # Sym expand
    model.expand_ncs(st, howtoname=howtoname)
    st.assemblies.clear()
    args.output_prfix += "_expanded"
    if args.pdb or args.cif:
        fileio.write_model(st, args.output_prfix, pdb=args.pdb, cif=args.cif)
    else:
        fileio.write_model(st, file_name=args.output_prfix+model_format)
# symmodel()

def symexpand(args):
    if args.chains: args.chains = sum(args.chains, [])
    model_format = fileio.check_model_format(args.model)

    howtoname = dict(dup=gemmi.HowToNameCopiedChain.Dup,
                     short=gemmi.HowToNameCopiedChain.Short,
                     number=gemmi.HowToNameCopiedChain.AddNumber)[args.howtoname]

    st = fileio.read_structure(args.model)

    if args.chains:
        logger.write("Keep {} chains only".format(" ".join(args.chains)))
        chains = set(args.chains)
        for m in st:
            to_del = [c.name for c in m if c.name not in chains]
            for c in to_del: m.remove_chain(c)

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    if not args.output_prfix:
        args.output_prfix = fileio.splitext(os.path.basename(args.model))[0]

    if len(st.ncs) > 0:
        symmetry.show_ncs_operators_axis_angle(st.ncs)
        non_given = [op for op in st.ncs if not op.given]
        if len(non_given) > 0:
            st_tmp = st.clone()
            model.expand_ncs(st_tmp, howtoname=howtoname)
            output_prfix = args.output_prfix + "_ncs_expanded"
            if args.pdb or args.cif:
                fileio.write_model(st_tmp, output_prfix, pdb=args.pdb, cif=args.cif)
            else:
                fileio.write_model(st_tmp, file_name=output_prfix+model_format)
        else:
            logger.write("All operators are already expanded (marked as given). Exiting.")
    else:
        logger.write("No NCS operators found. Exiting.")
    
    if len(st.assemblies) > 0: # should we support BIOMT?
        pass
# symexpand()

def h_add(args):
    st = fileio.read_structure(args.model)
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_h" + model_format
        logger.write("Output file: {}".format(args.output))
        
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(st,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)
    restraints.add_hydrogens(st, monlib, args.pos)
    fileio.write_model(st, file_name=args.output)
# h_add()

def merge_models(args):
    logger.write("Reading file   1: {}".format(args.models[0]))
    st = fileio.read_structure(args.models[0])
    logger.write("                  chains {}".format(" ".join([c.name for c in st[0]])))

    for i, f in enumerate(args.models[1:]):
        logger.write("Reading file {:3d}: {}".format(i+2, f))
        st2 = fileio.read_structure(f)
        for c in st2[0]:
            org_id = c.name
            c2 = st[0].add_chain(c, unique_name=True)
            if c.name != c2.name:
                logger.write("                  chain {} merged (ID changed to {})".format(c.name, c2.name))
            else:
                logger.write("                  chain {} merged".format(c.name))

    fileio.write_model(st, file_name=args.output)
# merge_models()

def show_power(args):
    maps_in = []
    if args.map:
        print(args.map)
        print(sum(args.map, []))
        maps_in = [(f,) for f in sum(args.map, [])]
        
    if args.halfmaps:
        args.halfmaps = sum(args.halfmaps, [])
        if len(args.halfmaps)%2 != 0:
            raise RuntimeError("Number of half maps is not even.")
        maps_in.extend([(args.halfmaps[2*i],args.halfmaps[2*i+1]) for i in range(len(args.halfmaps)//2)])
        
    if args.mask:
        mask = numpy.array(fileio.read_ccp4_map(args.mask)[0])
    else:
        mask = None

    hkldata = None
    labs = []
    for mapin in maps_in:
        tmp = maps.mask_and_fft_maps([fileio.read_ccp4_map(f) for f in mapin],
                                     args.resolution, mask)
        labs.append("F{:02d}".format(len(labs)+1))
        tmp.df.rename(columns=dict(FP=labs[-1]), inplace=True)
        if hkldata is None:
            hkldata = tmp
        else:
            if hkldata.cell != tmp.cell: raise RuntimeError("Different unit cell!")
            hkldata.merge(tmp.df[["H","K","L",labs[-1]]])

    if not labs:
        logger.write("No map files given. Exiting.")
        return
            
    hkldata.setup_relion_binning()
    bin_limits = dict(hkldata.bin_and_limits())

    ofs = open(args.output_prefix+".log", "w")
    ofs.write("Input:\n")
    for i in range(len(maps_in)):
        ofs.write("{} from {}\n".format(labs[i], " ".join(maps_in[i])))
    ofs.write("\n")
    
    ofs.write("""$TABLE: Power spectrum :
$GRAPHS
: log10(Mn(|F|^2)) :A:1,{}:
$$
1/resol^2 n d_max d_min {}
$$
$$
""".format(",".join([str(i+5) for i in range(len(labs))]), " ".join(labs)))
    print(hkldata.df)
    for i_bin, g in hkldata.binned():
        bin_d_max, bin_d_min = bin_limits[i_bin]
        ofs.write("{:.4f} {:7d} {:7.3f} {:7.3f}".format(1/bin_d_min**2, g[labs[0]].size, bin_d_max, bin_d_min,))
        for lab in labs:
            pwr = numpy.log10(numpy.average(numpy.abs(g[lab].to_numpy())**2))
            ofs.write(" {:.4e}".format(pwr))
        ofs.write("\n")
    ofs.write("$$\n")
# show_power()

def fcalc(args):
    if args.ligand: args.ligand = sum(args.ligand, [])
    if not args.output_prefix: args.output_prefix = fileio.splitext(os.path.basename(args.model))[0] + "_fcalc"

    st = fileio.read_structure(args.model)
    if args.cell is not None: st.cell = gemmi.UnitCell(*args.cell)
    if not st.cell.is_crystal():
        logger.error("ERROR: No unit cell information. Give --cell.")
        return
    monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                   stop_for_unknowns=False, check_hydrogen=True)
    fc_asu = model.calc_fc_fft(st, args.resolution, cutoff=args.cutoff, rate=args.rate,
                               monlib=monlib, source=args.source)
    mtz = gemmi.Mtz()
    mtz.spacegroup = fc_asu.spacegroup
    mtz.cell = fc_asu.unit_cell
    mtz.add_dataset('HKL_base')
    for label in ['H', 'K', 'L']: mtz.add_column(label, 'H')
    mtz.add_column("FC", "F")
    mtz.add_column("PHIC", "P")
    mtz.set_data(fc_asu)
    mtz.write_to_file(args.output_prefix+".mtz")
    logger.write("{} written.".format(args.output_prefix+".mtz"))
# fcalc()

def show(args):
    for filename in args.files:
        ext = fileio.splitext(filename)[1]
        if ext in (".mrc", ".ccp4", ".map"):
            fileio.read_ccp4_map(filename)
            logger.write("\n")
# show()

def json2csv(args):
    if not args.output_prefix:
        args.output_prefix = fileio.splitext(os.path.basename(args.json))[0]
        
    df = pandas.read_json(args.json)
    df.to_csv(args.output_prefix+".csv", index=False)
    logger.write("Output: {}".format(args.output_prefix+".csv"))
# json2csv()

def main(args):
    comms = dict(show=show,
                 json2csv=json2csv,
                 symmodel=symmodel,
                 expand=symexpand,
                 h_add=h_add,
                 merge_models=merge_models,
                 power=show_power,
                 fcalc=fcalc)
    
    com = args.subcommand
    f = comms.get(com)
    if f:
        return f(args)
    else:
        logger.error("Unknown subcommand: {}".format(com))
        return
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
