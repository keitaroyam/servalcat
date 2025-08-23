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
from servalcat.utils import hkl
from servalcat.utils import restraints
from servalcat.utils import maps
from servalcat.refmac import refmac_keywords
from servalcat.refine.refine import Geom, RefineParams
from servalcat import ext
import os
import gemmi
import numpy
import scipy.spatial
import pandas
import json
import re
import argparse

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
    group.add_argument('--cell', type=float, nargs=6, metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                       help="Box size")
    sym_group = parser.add_argument_group("symmetry")
    symmetry.add_symmetry_args(sym_group, require_pg=True)
    parser.add_argument('--contacting_only', action="store_true", help="Filter out non-contacting NCS")
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

    # helical_biomt
    parser = subparsers.add_parser("helical_biomt", description="generate BIOMT of helical reconstruction for PDB deposition")
    parser.add_argument('--model', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--map', help="Take box size from the map")
    group.add_argument('--cell', type=float, nargs=6, metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                       help="Box size")
    sym_group = parser.add_argument_group("symmetry")
    symmetry.add_symmetry_args(sym_group, require_pg=True)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--howtoname', choices=["dup", "short", "number"], default="short",
                        help="How to decide new chain IDs in expanded model (default: short); "
                        "dup: use original chain IDs (with different segment IDs), "
                        "short: use unique new IDs, "
                        "number: add number to original chain ID")
    parser.add_argument('-o', '--output_prfix')

    # expand
    parser = subparsers.add_parser("expand", description="Expand symmetry")
    parser.add_argument('--model', required=True)
    parser.add_argument('--chains', nargs="*", action="append", help="Select chains to keep")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--howtoname', choices=["dup", "short", "number"], default="short",
                       help="How to decide new chain IDs in expanded model (default: short); "
                       "dup: use original chain IDs (with different segment IDs), "
                       "short: use unique new IDs, "
                       "number: add number to original chain ID")
    group.add_argument("--split", action="store_true", help="split file for each operator")
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

    # add_op3
    parser = subparsers.add_parser("add_op3", description = "Add OP3 atoms to 5' ends")
    parser.add_argument('model')
    parser.add_argument('--chains', nargs="*", action="append", help="For selected chains only")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('-o','--output')

    # map_peaks
    parser = subparsers.add_parser("map_peaks", description = 'List density peaks and write a coot script')
    parser.add_argument('--model', required=True, help="Model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--map', help="Map file")
    group.add_argument('--mtz', help="MTZ for map file")
    parser.add_argument('--mtz_labels', default="DELFWT,PHDELWT", help='F,PHI labels (default: %(default)s)')
    parser.add_argument('--oversample_pixel', type=float, help='Desired pixel spacing in map (Angstrom)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sigma_level', type=float, help="Threshold map level in sigma unit")
    group.add_argument('--abs_level', type=float, help="Threshold map level in absolute unit")
    parser.add_argument('--blob_pos', choices=["peak", "centroid"], default="centroid",
                       help="default: %(default)s")
    parser.add_argument('--min_volume', type=float, default=0.3, help="minimum blob volume (default: %(default).1f)")
    parser.add_argument('--max_volume', type=float, help="maximum blob volume (default: none)")
    parser.add_argument('-o','--output_prefix', default="peaks")
    
    # h_density
    parser = subparsers.add_parser("h_density", description = 'Hydrogen density analysis')
    parser.add_argument('--model', required=True, help="Model with hydrogen atoms")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--map', help="Fo-Fc map file")
    group.add_argument('--mtz', help="MTZ for Fo-Fc map file")
    parser.add_argument('--mtz_labels', default="DELFWT,PHDELWT", help='F,PHI labels (default: %(default)s)')
    parser.add_argument('--oversample_pixel', type=float, help='Desired pixel spacing in map (Angstrom)')
    #parser.add_argument("--source", choices=["electron", "xray", "neutron"], default="electron")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sigma_level', type=float, help="Threshold map level in sigma unit")
    group.add_argument('--abs_level', type=float, help="Threshold map level in absolute unit")
    parser.add_argument('--max_dist', type=float, default=0.5, help="max distance between peak and hydrogen position in the model (default: %(default).1f)")
    parser.add_argument('--blob_pos', choices=["peak", "centroid"], default="centroid",
                       help="default: %(default)s")
    parser.add_argument('--min_volume', type=float, default=0.3, help="minimum blob volume (default: %(default).1f)")
    parser.add_argument('--max_volume', type=float, default=3, help="maximum blob volume (default: %(default).1f)")
    parser.add_argument('-o','--output_prefix')

    # fix_link
    parser = subparsers.add_parser("fix_link", description = 'Fix LINKR/_struct_conn records in the model')
    parser.add_argument('model')
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--bond_margin', type=float, default=1.3, help='(default: %(default).1f)')
    parser.add_argument('--metal_margin', type=float, default=1.1, help='(default: %(default).1f)')
    parser.add_argument('-o','--output', help="Default: input_fixlink.{pdb|mmcif}")

    # merge_models
    parser = subparsers.add_parser("merge_models", description = 'Merge multiple model files')
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o','--output', required=True)

    # merge_dicts
    parser = subparsers.add_parser("merge_dicts", description = 'Merge restraint dictionary cif files')
    parser.add_argument('cifs', nargs="+")
    parser.add_argument('-o','--output', default="merged.cif", help="Output cif file (default: %(default)s)")

    # geom
    parser = subparsers.add_parser("geom", description = 'Calculate geometry and show outliers')
    parser.add_argument('model')
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--keywords', nargs='+', action="append",
                        help="refmac keyword(s)")
    parser.add_argument('--keyword_file', nargs='+', action="append",
                        help="refmac keyword file(s)")
    parser.add_argument('--sigma', type=float, default=5,
                        help="sigma cutoff to print outliers (default: %(default).1f)")
    parser.add_argument('--per_atom_score_as_b', action='store_true',
                        help="write model file with per-atom score as B values")
    parser.add_argument("--check_skew", action='store_true', help="(experimental) check bond skew to test magnification")
    parser.add_argument('-n', '--nucleus', action="store_true", help="Use nucleus distances (for neutron)")
    parser.add_argument("--ignore_h", action='store_true', help="ignore hydrogen")
    parser.add_argument("--selection", help="evaluate part of the model")
    parser.add_argument('-o', '--output_prefix', 
                        help="default: taken from input file")

    # conf
    parser = subparsers.add_parser("conf", description = 'Compare conformations')
    parser.add_argument('models', nargs="+")
    parser.add_argument("--min_diff", type=float, default=60.)
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument("--same_chain", action='store_true', help="Only between same chains (more than one file)")
    parser.add_argument('-o', '--output_prefix', default="conf",
                        help="")

    # adp
    parser = subparsers.add_parser("adp", description = 'ADP analysis')
    parser.add_argument('model')
    parser.add_argument('-o', '--output_prefix',
                        help="default: taken from input file")
    
    # power
    parser = subparsers.add_parser("power", description = 'Show power spectrum')
    parser.add_argument("--map",  nargs="*", action="append")
    parser.add_argument("--halfmaps",  nargs="*", action="append")
    parser.add_argument('--mask', help='Mask file')
    parser.add_argument('-d', '--resolution', type=float)
    parser.add_argument('-o', '--output_prefix', default="power")

    # fcalc
    parser = subparsers.add_parser("fcalc", description = 'Structure factor from model')
    parser.add_argument('--model', required=True)
    parser.add_argument("--no_expand_ncs", action='store_true', help="Do not expand strict NCS in MTRIX or _struct_ncs_oper")
    parser.add_argument("--method", choices=["fft", "direct"], default="fft")
    parser.add_argument("--source", choices=["electron", "xray", "neutron", "custom"], default="electron")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--cell', type=float, nargs=6, metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                        help="Override unit cell")
    parser.add_argument('--auto_box_with_padding', type=float, help="Determine box size from model with specified padding")
    parser.add_argument('--cutoff', type=float, default=1e-5)
    parser.add_argument('--rate', type=float, default=1.5)
    parser.add_argument('--add_dummy_sigma', action='store_true', help="write dummy SIGF")
    parser.add_argument('--as_intensity', action='store_true', help="if you want |F|^2")
    parser.add_argument('--keep_charges',  action='store_true',
                        help="Use scattering factor for charged atoms. Use it with care.")
    parser.add_argument('-d', '--resolution', type=float, required=True)
    parser.add_argument('-o', '--output_prefix')

    # nemap
    parser = subparsers.add_parser("nemap", description = 'Normalized expected map calculation from half maps')
    parser.add_argument("--halfmaps", required=True, nargs=2)
    parser.add_argument('--pixel_size', type=float, help='Override pixel size (A)')
    parser.add_argument("--half1_only", action='store_true', help="Only use half 1 for map calculation (use half 2 only for noise estimation)")
    parser.add_argument('-B', type=float, help="local B value")
    parser.add_argument("--no_fsc_weights", action='store_true',
                        help="Just for debugging purpose: turn off FSC-based weighting")
    parser.add_argument("--sharpening_b", type=float,
                        help="Use B value (negative value for sharpening) instead of standard deviation of the signal")
    parser.add_argument("-d", '--resolution', type=float)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-o', '--output_prefix', default='nemap')
    parser.add_argument("--trim", action='store_true', help="Write trimmed maps")
    parser.add_argument("--trim_mtz", action='store_true', help="Write trimmed mtz")
    parser.add_argument("--local_fourier_weighting_with", type=float, default=0,
                        help="Experimental: give kernel size in A^-1 unit to use local Fourier weighting instead of resolution-dependent weights")

    # blur
    parser = subparsers.add_parser("blur", description = 'Blur data by specified B value')
    parser.add_argument('--hklin', required=True, help="input MTZ file")
    parser.add_argument('-B', type=float, required=True, help="B value for blurring (negative value for sharpening)")
    parser.add_argument('-o', '--output_prefix')

    # mask_from_model
    parser = subparsers.add_parser("mask_from_model", description = 'Make a mask from model')
    parser.add_argument("--map", required=True, help="For unit cell and pixel size reference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection")
    parser.add_argument('--radius', type=float, required=True,
                        help='Radius in angstrom')
    parser.add_argument('--soft_edge', type=float, default=0,
                        help='Soft edge (default: %(default).1f)')
    parser.add_argument('-o', '--output', default="mask_from_model.mrc")
    
    # applymask (and normalize within mask)
    parser = subparsers.add_parser("applymask", description = 'Apply mask and optionally normalize map within mask')
    parser.add_argument("--map", required=True)
    parser.add_argument('--mask', required=True, help='Mask file')
    parser.add_argument("--normalize", action='store_true',
                        help="Normalize map values using mean and sd within the mask")
    parser.add_argument("--trim", action='store_true', help="Write trimmed map")
    parser.add_argument('--mask_cutoff', type=float, default=0.5,
                        help="cutoff value for normalization and trimming (default: %(default)s)")
    parser.add_argument('-o', '--output_prefix')

    # map2mtz
    parser = subparsers.add_parser("map2mtz", description = 'FFT map and write an mtz')
    parser.add_argument("--map", required=True)
    parser.add_argument("-d", '--resolution', type=float)
    parser.add_argument('-o', '--output')

    # sm2mm
    parser = subparsers.add_parser("sm2mm", description = 'Small molecule files (cif/hkl/res/ins) to macromolecules (pdb/mmcif/mtz)')
    parser.add_argument('files', nargs='+', help='Cif/ins/res/hkl files')
    parser.add_argument('-o', '--output_prefix')

    # mm2ins
    parser = subparsers.add_parser("mm2ins", description = 'convert pdb/mmcif to ins for shelxl/olex2')
    parser.add_argument('model')
    parser.add_argument('-o', '--output')

    # seq
    parser = subparsers.add_parser("seq", description = 'Print/align model sequence')
    parser.add_argument("--model", required=True)
    parser.add_argument('--seq', nargs="*", action="append", help="Sequence file(s)")
    parser.add_argument('--scoring', nargs=6, type=int, default=(1, 0, -1, -1, 0, -1),
                        metavar=("match", "mismatch", "gapo", "gape", "good_gapo", "bad_gapo"),
                        help="scoring function. default: %(default)s")

    # dnarna
    parser = subparsers.add_parser("dnarna", description = 'DNA to RNA or RNA to DNA model conversion')
    parser.add_argument("model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to_dna', action='store_true', help="To DNA")
    group.add_argument('--to_rna', action='store_true', help="To RNA")
    parser.add_argument('--chains', nargs="*", action="append", help="Select chains to convert")
    parser.add_argument('-o', '--output')

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
        raise SystemExit("ERROR: give both helical parameters --twist and --rise")

    is_helical = args.twist is not None
    st, cif_ref = fileio.read_structure_from_pdb_and_mmcif(args.model)
    st.spacegroup_hm = "P 1"
    map_and_start = None
    if args.map:
        logger.writeln("Reading cell from map")
        map_and_start = fileio.read_ccp4_map(args.map, header_only=True)
        st.cell = map_and_start[0].unit_cell
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        raise SystemExit("Error: Unit cell parameters look wrong. Please use --map or --cell")

    if args.chains:
        logger.writeln("Keep {} chains only".format(" ".join(args.chains)))
        chains = set(args.chains)
        for m in st:
            to_del = [c.name for c in m if c.name not in chains]
            for c in to_del: m.remove_chain(c)
        if st[0].count_atom_sites() == 0:
            raise SystemExit("ERROR: no atoms left. Check --chains option.")

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    symmetry.update_ncs_from_args(args, st, map_and_start=map_and_start, filter_contacting=args.contacting_only)

    if args.biomt:
        st.assemblies.clear()
        st.raw_remarks = []
        a = model.prepare_assembly("1", all_chains, st.ncs, is_helical=is_helical)
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

def helical_biomt(args):
    if (args.twist, args.rise).count(None) > 0:
        raise SystemExit("ERROR: give helical parameters --twist and --rise")

    model_format = fileio.check_model_format(args.model)
    howtoname = dict(dup=gemmi.HowToNameCopiedChain.Dup,
                     short=gemmi.HowToNameCopiedChain.Short,
                     number=gemmi.HowToNameCopiedChain.AddNumber)[args.howtoname]

    st, cif_ref = fileio.read_structure_from_pdb_and_mmcif(args.model)
    st.spacegroup_hm = "P 1"
    map_and_start = None
    if args.map:
        logger.writeln("Reading cell from map")
        map_and_start = fileio.read_ccp4_map(args.map, header_only=True)
        st.cell = map_and_start[0].unit_cell
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        raise SystemExit("Error: Unit cell parameters look wrong. Please use --map or --cell")

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    ncsops = symmetry.ncsops_from_args(args, st.cell, map_and_start=map_and_start, st=st,
                                       helical_min_n=args.start, helical_max_n=args.end)
    #ncsops = [x for x in ncsops if not x.tr.is_identity()] # remove identity

    logger.writeln("")
    logger.writeln("-------------------------------------------------------------")
    logger.writeln("You may need to write following matrices in OneDep interface:")
    for idx, op in enumerate(ncsops):
        logger.writeln("")
        logger.writeln("operator {}".format(idx+1))
        mat = op.tr.mat.tolist()
        vec = op.tr.vec.tolist()
        for i in range(3):
            mstr = ["{:10.6f}".format(mat[i][j]) for j in range(3)]
            logger.writeln("{} {:14.5f}".format(" ".join(mstr), vec[i]))
    logger.writeln("-------------------------------------------------------------")
    logger.writeln("")

    # BIOMT
    st.assemblies.clear()
    st.raw_remarks = []
    a = model.prepare_assembly("1", all_chains, ncsops, is_helical=True)
    st.assemblies.append(a)

    if not args.output_prfix:
        args.output_prfix = fileio.splitext(os.path.basename(args.model))[0] + "_biomt"

    fileio.write_model(st, args.output_prfix, pdb=(model_format == ".pdb"), cif=True, cif_ref=cif_ref)
    logger.writeln("")
    logger.writeln("These {}.* files may be used for deposition (once OneDep implemented reading BIOMT from file..)".format(args.output_prfix))
    logger.writeln("")
    # BIOMT expand
    st.transform_to_assembly("1", howtoname)
    args.output_prfix += "_expanded"
    fileio.write_model(st, file_name=args.output_prfix+model_format)
    logger.writeln(" note that this expanded model file is just for visual inspection, *not* for deposition!")
# helical_biomt()

def symexpand(args):
    if args.chains: args.chains = sum(args.chains, [])
    model_format = fileio.check_model_format(args.model)
    if not args.split:
        howtoname = dict(dup=gemmi.HowToNameCopiedChain.Dup,
                         short=gemmi.HowToNameCopiedChain.Short,
                         number=gemmi.HowToNameCopiedChain.AddNumber)[args.howtoname]

    st = fileio.read_structure(args.model)

    if args.chains:
        logger.writeln("Keep {} chains only".format(" ".join(args.chains)))
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
            if args.split:
                for i, op in enumerate(st.ncs):
                    if op.given: continue
                    st_tmp = st.clone()
                    for m in st_tmp: m.transform_pos_and_adp(op.tr)
                    output_prfix = args.output_prfix + "_ncs_{:02d}".format(i+1)
                    if args.pdb or args.cif:
                        fileio.write_model(st_tmp, output_prfix, pdb=args.pdb, cif=args.cif)
                    else:
                        fileio.write_model(st_tmp, file_name=output_prfix+model_format)
            else:
                st_tmp = st.clone()
                model.expand_ncs(st_tmp, howtoname=howtoname)
                output_prfix = args.output_prfix + "_ncs_expanded"
                if args.pdb or args.cif:
                    fileio.write_model(st_tmp, output_prfix, pdb=args.pdb, cif=args.cif)
                else:
                    fileio.write_model(st_tmp, file_name=output_prfix+model_format)
        else:
            logger.writeln("All operators are already expanded (marked as given). Exiting.")
    else:
        logger.writeln("No NCS operators found. Exiting.")
    
    if len(st.assemblies) > 0: # should we support BIOMT?
        pass
# symexpand()

def h_add(args):
    st = fileio.read_structure(args.model)
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_h" + model_format
    logger.writeln("Output file: {}".format(args.output))
        
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(st,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)
    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    restraints.find_and_fix_links(st, monlib, find_metal_links=False, add_found=False)
    try:
        restraints.add_hydrogens(st, monlib, args.pos)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    fileio.write_model(st, file_name=args.output)
# h_add()

def add_op3(args):
    if args.chains: args.chains = sum(args.chains, [])
    st = fileio.read_structure(args.model)
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_op3" + model_format
    logger.writeln("Output file: {}".format(args.output))
    
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(st,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)
    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)

    for chain in st[0]:
        if args.chains and chain.name not in args.chains: continue
        p = chain.get_polymer()
        if not p: continue
        p_type = p.check_polymer_type()
        if p_type not in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna): continue
        r0 = p[0]
        # TODO: alias
        # TODO: altlocs
        alt = "*"
        if r0.find_atom("OP3", alt): continue
        a_op1 = r0.find_atom("OP1", alt)
        a_op2 = r0.find_atom("OP2", alt)
        a_o5p = r0.find_atom("O5'", alt)
        a_p = r0.find_atom("P", alt)
        if None in (a_op1, a_op2, a_o5p, a_p):
            logger.writeln(f"Error: atoms not found. skipping {chain.name}/{r0}")
            continue
        logger.writeln(f"Adding OP3 to {chain.name}/{r0}")
        a_op3 = r0.add_atom(a_p) # inherit ADP and occupancy
        a_op3.name = "OP3"
        a_op3.element = gemmi.Element("O")
        v1 = a_p.pos - a_op1.pos
        v2 = a_p.pos - a_op2.pos
        v3 = a_p.pos - a_o5p.pos
        v = v1 + v2 + v3
        a_op3.pos = a_p.pos + v / v.length() * 1.517
    
    fileio.write_model(st, file_name=args.output)
# add_op3()

def read_map_and_oversample(map_in=None, mtz_in=None, mtz_labs=None, oversample_pixel=None):
    if mtz_in is not None:
        mtz = fileio.read_mmhkl(mtz_in)
        lab_f, lab_phi = mtz_labs.split(",")
        asu = mtz.get_f_phi(lab_f, lab_phi)
        if oversample_pixel is not None:
            d_min = numpy.min(asu.make_d_array())
            sample_rate = d_min / oversample_pixel
        else:
            sample_rate = 3
        gr = asu.transform_f_phi_to_map(sample_rate=sample_rate)
    elif map_in is not None:
        gr = fileio.read_ccp4_map(map_in)[0]
        if oversample_pixel is not None:
            asu = gemmi.transform_map_to_f_phi(gr).prepare_asu_data()
            d_min = numpy.min(asu.make_d_array())
            sample_rate = d_min / oversample_pixel
            gr = asu.transform_f_phi_to_map(sample_rate=sample_rate)
    else:
        raise SystemExit("Invalid input")
    
    if oversample_pixel is not None:
        logger.writeln("--oversample_pixel= {} is requested.".format(oversample_pixel))
        logger.writeln(" recalculated grid:")
        logger.writeln("  {:4d} {:4d} {:4d}".format(*gr.shape))
        logger.writeln(" spacings:")
        logger.writeln("  {:.6f} {:.6f} {:.6f}".format(*gr.spacing))
        #maps.write_ccp4_map("{}_oversampled.mrc".format(output_prefix), gr)

    return gr
# read_map_and_oversample()    

def map_peaks(args):
    st = fileio.read_structure(args.model)
    gr = read_map_and_oversample(map_in=args.map, mtz_in=args.mtz, mtz_labs=args.mtz_labels,
                                 oversample_pixel=args.oversample_pixel)
    gr_sigma = numpy.std(gr)
    if args.abs_level is not None:
        cutoff = args.abs_level
    else:
        cutoff = args.sigma_level * gr_sigma # assuming mean(gr) = 0
        
    blobs = gemmi.find_blobs_by_flood_fill(gr, cutoff,
                                           min_volume=args.min_volume, min_score=0)
    blobs.extend(gemmi.find_blobs_by_flood_fill(gr, cutoff, negate=True,
                                                min_volume=args.min_volume, min_score=0))
    getpos = dict(peak=lambda x: x.peak_pos,
                  centroid=lambda x: x.centroid)[args.blob_pos]
    st_peaks = model.st_from_positions([getpos(b) for b in blobs])
    st_peaks.cell = st.cell
    st_peaks.ncs = st.ncs
    st_peaks.setup_cell_images()
    logger.writeln("{} peaks detected".format(len(blobs)))
    #st_peaks.write_pdb("peaks.pdb")
    
    # Filter symmetry related
    ns = gemmi.NeighborSearch(st_peaks[0], st_peaks.cell, 5.).populate()
    cs = gemmi.ContactSearch(1.)
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    results = cs.find_contacts(ns)
    del_idxes = set()
    for r in results:
        if r.partner1.residue.seqid.num not in del_idxes:
            del_idxes.add(r.partner2.residue.seqid.num)
    for i in reversed(sorted(del_idxes)):
        del st_peaks[0][0][i]
        del blobs[i]
    #st_peaks.write_pdb("peaks_asu.pdb")
    logger.writeln("{} peaks after removing symmetry equivalents".format(len(blobs)))

    # Assign to nearest atom
    ns = gemmi.NeighborSearch(st[0], st.cell, 10.).populate() # blob is rejected if > 10 A. ok?
    peaks = []
    for b in blobs:
        bpos = getpos(b)
        map_val = gr.interpolate_value(bpos)
        if (args.max_volume is not None and b.volume > args.max_volume) or abs(map_val) < cutoff: continue
        x = ns.find_nearest_atom(bpos)
        if x is None: # this should not happen
            logger.writeln("no nearest atom: value={:.2e} volume= {:.2f} pos= {}".format(map_val, b.volume, bpos))
            continue
        chain = st[0][x.chain_idx]
        res = chain[x.residue_idx]
        atom = res[x.atom_idx]
        im = st.cell.find_nearest_image(atom.pos, bpos, gemmi.Asu.Any)
        mpos = st.cell.find_nearest_pbc_position(atom.pos, bpos, im.sym_idx)
        dist = atom.pos.dist(mpos)
        peaks.append((map_val, b.volume, mpos, dist, chain, res, atom))

    if len(peaks) == 0:
        logger.writeln("No peaks found. Change parameter(s).")
        return
        
    # Print and write coot script
    peaks.sort(reverse=True, key=lambda x:(abs(x[0]), x[1]))
    for_coot = []
    for_df = []
    for i, p in enumerate(peaks):
        map_val, volume, mpos, dist, chain, res, atom = p
        mpos_str = "({: 7.2f},{: 7.2f},{: 7.2f})".format(mpos.x, mpos.y, mpos.z)
        atom_name = atom.name + ("." + atom.altloc if atom.altloc != "\0" else "")
        atom_str = "{}/{}/{}".format(chain.name, res.seqid, atom_name)
        if args.abs_level is None:
            map_val /= gr_sigma
        lab_str = "Peak {:4d} value= {: .2e} volume= {:5.1f} pos= {} closest= {:10s} dist= {:.2f}".format(i+1, map_val, volume, mpos_str, atom_str, dist)
        for_coot.append((lab_str, (mpos.x, mpos.y, mpos.z)))
        for_df.append((map_val, volume, mpos.x, mpos.y, mpos.z, chain.name, str(res.seqid), atom_name, dist))
    df = pandas.DataFrame(for_df, columns=["map_value" if args.abs_level is not None else "sigma_level",
                                           "volume", "x", "y", "z", "chain", "residue", "atom", "dist"])
    logger.writeln(df.to_string())
    with open(args.output_prefix + ".json", "w") as ofs:
        df.to_json(ofs, orient="records", indent=2)
        logger.writeln("saved: {}".format(ofs.name))
    coot_out = args.output_prefix + "_coot.py"
    with open(coot_out, "w") as ofs:
        ofs.write("""\
from __future__ import absolute_import, division, print_function
import gtk
class coot_serval_map_peak_list:
  def __init__(self):
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.set_title("Map peaks (Servalcat)")
    window.set_default_size(600, 600)
    scrolled_win = gtk.ScrolledWindow()
    scrolled_win.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
    vbox = gtk.VBox(False, 2)
    frame_vbox = gtk.VBox(False, 0)
    frame_vbox.set_border_width(3)
    self.btns = []
    self.data = {}
    self.add_data(frame_vbox)
    scrolled_win.add_with_viewport(frame_vbox)
    vbox.pack_start(scrolled_win, True, True, 0)
    window.add(vbox)
    window.show_all()
    self.toggled(self.btns[0], 0)

  def toggled(self, btn, i):
    if btn.get_active():
      set_rotation_centre(*self.data[i][1])
      add_status_bar_text(self.data[i][0])

  def add_data(self, vbox):
    for i, d in enumerate(self.data):
      self.btns.append(gtk.RadioButton(None if i == 0 else self.btns[0], d[0]))
      vbox.pack_start(self.btns[-1], False, False, 0)
      self.btns[-1].connect('toggled', self.toggled, i)

gui = coot_serval_map_peak_list()
""".format(for_coot))
    logger.writeln("\nRun:")
    logger.writeln("coot --script {}".format(coot_out))
# map_peaks()

def h_density_analysis(args):
    #if args.source != "electron":
    #    raise SystemExit("Only electron source is supported.")
    model_format = fileio.check_model_format(args.model)
    st = fileio.read_structure(args.model)
    if not st[0].has_hydrogen():
        raise SystemExit("No hydrogen in model.")

    if args.output_prefix is None:
        args.output_prefix = fileio.splitext(os.path.basename(args.model))[0] + "_hana"

    gr = read_map_and_oversample(map_in=args.map, mtz_in=args.mtz, mtz_labs=args.mtz_labels,
                                 oversample_pixel=args.oversample_pixel)
                            
    if args.abs_level is not None:
        cutoff = args.abs_level
    else:
        cutoff = args.sigma_level * numpy.std(gr) # assuming mean(gr) = 0

    blobs = gemmi.find_blobs_by_flood_fill(gr, cutoff,
                                           min_volume=args.min_volume, min_score=0)
    getpos = dict(peak=lambda x: x.peak_pos,
                  centroid=lambda x: x.centroid)[args.blob_pos]

    peaks = [getpos(b).tolist() for b in blobs]
    kdtree = scipy.spatial.cKDTree(peaks)
    found = []
    n_hydr = 0
    h_assigned = [0 for _ in range(len(blobs))]
    st2 = st.clone()
    for ic, chain in enumerate(st[0]):
        for ir, res in enumerate(chain):
            for ia, atom in reversed(list(enumerate(res))):
                if not atom.is_hydrogen(): continue
                n_hydr += 1
                dist, idx = kdtree.query(atom.pos.tolist(), k=1, p=2)
                map_val = gr.interpolate_value(getpos(blobs[idx]))
                if dist < args.max_dist and blobs[idx].volume < args.max_volume and map_val > cutoff:
                    found.append((getpos(blobs[idx]), map_val, dist,  blobs[idx].volume, 
                                  chain.name, str(res.seqid), res.name,
                                  atom.name, atom.altloc.replace("\0","")))
                    h_assigned[idx] = 1
                else:
                    del st2[0][ic][ir][ia]

    found.sort(key=lambda x: x[1], reverse=True)
    logger.writeln("")
    logger.writeln("Found hydrogen peaks:")
    logger.writeln("dist map  vol  atom")
    for _, map_val, dist, volume, chain, resi, resn, atom, alt in found:
        logger.writeln("{:.2f} {:.2f} {:.2f} {}/{} {}/{}{}".format(dist, map_val, volume,
                                                                 chain, resn, resi,
                                                                 atom, "."+alt if alt else ""))

    logger.writeln("")
    logger.writeln("Result:")
    logger.writeln(" number of hydrogen in the model  : {}".format(n_hydr))
    logger.writeln(" number of peaks close to hydrogen: {} ({:.1%})".format(len(found), len(found)/n_hydr))
    logger.writeln("")

    st_peaks = model.st_from_positions([getpos(b) for b in blobs],
                                       bs=[gr.interpolate_value(getpos(b)) for b in blobs],
                                       qs=h_assigned)
    fileio.write_model(st_peaks, file_name="{}_peaks.mmcif".format(args.output_prefix))
    logger.writeln(" this file includes peak positions")
    logger.writeln(" occ=1: hydrogen assigned, occ=0: unassigned.")
    logger.writeln(" B: density value at {}".format(args.blob_pos))
    logger.writeln("")
    
    fileio.write_model(st2, file_name="{}_h_with_peak{}".format(args.output_prefix, model_format))
    logger.writeln(" this file is a copy of input model, where hydrogen atoms without peaks are removed.")
# h_density_analysis()

def fix_link(args):
    st = fileio.read_structure(args.model)
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_fixlink" + model_format
    logger.writeln("Output file: {}".format(args.output))
        
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(st,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)
    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    restraints.find_and_fix_links(st, monlib, bond_margin=args.bond_margin,
                                  metal_margin=args.metal_margin)
    fileio.write_model(st, file_name=args.output)
# fix_link()
    
def merge_models(args):
    logger.writeln("Reading file   1: {}".format(args.models[0]))
    st = fileio.read_structure(args.models[0])
    logger.writeln("                  chains {}".format(" ".join([c.name for c in st[0]])))

    for i, f in enumerate(args.models[1:]):
        logger.writeln("Reading file {:3d}: {}".format(i+2, f))
        st2 = fileio.read_structure(f)
        for c in st2[0]:
            org_id = c.name
            c2 = st[0].add_chain(c, unique_name=True)
            if c.name != c2.name:
                logger.writeln("                  chain {} merged (ID changed to {})".format(c.name, c2.name))
            else:
                logger.writeln("                  chain {} merged".format(c.name))

    fileio.write_model(st, file_name=args.output)
# merge_models()

def merge_dicts(args):
    fileio.merge_ligand_cif(args.cifs, args.output)
# merge_dicts()

def geometry(args):
    if args.ligand: args.ligand = sum(args.ligand, [])
    if not args.output_prefix: args.output_prefix = fileio.splitext(os.path.basename(args.model))[0] + "_geom"
    keywords = []
    if args.keywords or args.keyword_file:
        if args.keywords: keywords = sum(args.keywords, [])
        if args.keyword_file: keywords.extend(l for f in sum(args.keyword_file, []) for l in open(f))
    params = refmac_keywords.parse_keywords(keywords)
    st = fileio.read_structure(args.model)
    if args.ignore_h:
        st.remove_hydrogens()
    try:
        monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                 stop_for_unknowns=True, params=params)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    restraints.find_and_fix_links(st, monlib)
    try:
        topo, _ = restraints.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.NoChange,
                                              check_hydrogen=True, params=params)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    
    if args.selection:
        sel = gemmi.Selection(args.selection)
        atom_pos = [-1 for _ in range(st[0].count_atom_sites())]
        n = 0
        for chain in sel.chains(st[0]):
            for res in sel.residues(chain):
                for atom in sel.atoms(res):
                    atom_pos[atom.serial-1] = n
                    n += 1
        logger.writeln("Using selection '{}': {} atoms out of {}".format(args.selection, n, len(atom_pos)))
    else:
        atom_pos = None

    refine_params = RefineParams(st, refine_xyz=True)
    geom = Geom(st, topo, monlib, refine_params,
                params=params, atom_pos=atom_pos, use_nucleus=args.nucleus)
    for k in geom.outlier_sigmas: geom.outlier_sigmas[k] = args.sigma
    geom.setup_nonbonded()
    ret = geom.show_model_stats()
    
    with open(args.output_prefix + "_summary.json", "w") as ofs:
        ret["summary"].to_json(ofs, indent=2)
        logger.writeln("saved: {}".format(ofs.name))
    with open(args.output_prefix + "_outliers.json", "w") as ofs:
        for k in ret["outliers"]:
            ret["outliers"][k] = ret["outliers"][k].to_dict(orient="records")
        json.dump(ret["outliers"], ofs, indent=2)
        logger.writeln("saved: {}".format(ofs.name))

    if args.check_skew:
        logger.writeln("\nChecking skewness of bond length deviation")
        # better to ignore hydrogen
        tab = geom.geom.reporting.get_bond_outliers(use_nucleus=geom.use_nucleus, min_z=0)
        for a in "atom1", "atom2":
            tab[a] = [str(geom.lookup[x]) for x in tab[a]]
        df = pandas.DataFrame(tab)
        df["dev"] = df["value"] - df["ideal"]
        df = df.reindex(df.dev.abs().sort_values(ascending=False).index)
        logger.writeln("Bond length deviations:")
        logger.writeln(df.to_string(max_rows=20))
        q1, q2, q3 = numpy.percentile(df["dev"], [25, 50, 75])
        sk2 = (q1 + q3 - 2 * q2) / (q3 - q1)
        logger.writeln("bond_dev_median= {:.6f}".format(q2))
        logger.writeln("bond_dev_skew=   {:.4f}".format(df["dev"].skew()))
        logger.writeln("bond_dev_sk2=    {:.4f}".format(sk2))
        with open(args.output_prefix + "_bond_dev.html", "w") as ofs:
            ofs.write("""\
<html>
<head>
 <meta charset="utf-8" />
 <script src="https://cdn.plot.ly/plotly-2.20.0.min.js" charset="utf-8"></script>
</head>
<body>
 <div id="hist"></div>
 <script>
  var trace = {
   x: %s,
   type: 'histogram'
  };
  var layout = {
   title: "median: %.4f, sk2: %.4f",
   xaxis: {title: "bond distance - ideal"},
   yaxis: {title: "count"},
   shapes: [{
    type: 'line',
    yref: 'paper',
    x0: 0, y0: 0,
    x1: 0, y1: 1}]
  };
  target = document.getElementById('hist');
  Plotly.newPlot(target, [trace], layout);
 </script>
</body>
</html>
""" % (str(list(df.dev)), q2, sk2))
            logger.writeln("check histogram: {}".format(ofs.name))

    # Note that this modifies st
    if args.per_atom_score_as_b:
        model_format = fileio.check_model_format(args.model)
        peratom = geom.geom.reporting.per_atom_score(len(geom.atoms), geom.use_nucleus, "mean")
        for i, score in enumerate(peratom["total"]):
            geom.atoms[i].b_iso = score
        fileio.write_model(st, file_name="{}_per_atom_score{}".format(args.output_prefix, model_format))
# geometry()

def compare_conf(args):
    def angle_abs_diff(a, b, full=360.):
        # from gemmi/math.hpp
        d = abs(a - b)
        if d > full:
            d -= numpy.floor(d / full) * full
        return min(d, full - d)
    # angle_abs_diff()
    
    if args.ligand: args.ligand = sum(args.ligand, [])
    st = None
    for i, f in enumerate(args.models):
        tmp = fileio.read_structure(f)
        if len(args.models) > 1:
            for chain in tmp[0]:
                chain.name = f"{i+1}_{chain.name}"
        if i == 0:
            st = tmp
        else:
            for chain in tmp[0]:
                st[0].add_chain(chain)
    try:
        monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                 stop_for_unknowns=True)
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")

    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
    try:
        topo, _ = restraints.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.NoChange,
                                              check_hydrogen=False)
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
    ncslist = restraints.prepare_ncs_restraints(st)
    lookup = {x.atom: x for x in st[0].all()}
    ptypes = {x.name: x.polymer_type for x in st.entities}
    resn_lookup = {(chain.name, res.seqid): res.name for chain in st[0] for res in chain}
    confs = {}
    for t in topo.torsions:
        cra = lookup[t.atoms[0]]
        ptype = ptypes[cra.residue.entity_id]
        is_peptide = ptype in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD)
        is_peptide_tors = t.restr.label.startswith("chi") or t.restr.label in ("omega", "phi", "psi")
        is_na = ptype in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid)
        is_na_tors = t.restr.label in ("C2e-chi", "alpha", "beta", "gamma", "C2e-nyu0", "epsilon", "zeta")
        if (is_peptide and is_peptide_tors) or (is_na and is_na_tors):
            confs.setdefault(cra.chain.name, {}).setdefault(cra.residue.seqid, {})[t.restr.label] = numpy.rad2deg(t.calculate())
    fulls = {("ARG", "chi5"): 180., ("TYR", "chi2"): 180., ("PHE", "chi2"): 180., ("ASP", "chi2"): 180., ("GLU", "chi3"): 180.}
    ret = []
    for_coot = []
    for ncs in ncslist.ncss:
        c1, c2 = ncs.chains
        if args.same_chain and len(args.models) > 1 and c1[c1.index("_"):] != c2[c2.index("_"):]:
            continue
        for s1, s2 in ncs.seqids:
            if c1 in confs and s1 in confs[c1] and c2 in confs and s2 in confs[c2]:
                conf1, conf2 = confs[c1][s1], confs[c2][s2]
                resn = resn_lookup[(c1, s1)]
                for t in conf1:
                    if t in conf2:
                        d = float(angle_abs_diff(conf1[t], conf2[t], fulls.get((resn, t), 360.)))
                        ret.append((c1, s1, c2, s2, resn, t, conf1[t], conf2[t], d))
                        if d > args.min_diff:
                            for_coot.append((c1, s1.num, c2, s2.num, resn, t, d))
    df = pandas.DataFrame(ret, columns=["chain_1", "seq_1", "chain_2", "seq_2", "resn", "label", "conf_1", "conf_2", "diff"])
    df.sort_values("diff", ascending=False, inplace=True)
    logger.writeln(f"\nList of torsion angle differences (>{args.min_diff})")
    logger.writeln(df[df["diff"] > args.min_diff].to_string(index=False))

    for_coot.sort(key=lambda x:-x[-1])
    coot_out = args.output_prefix + "_coot.py"
    with open(coot_out, "w") as ofs:
        # https://python-gtk-3-tutorial.readthedocs.io/en/latest/treeview.html
        ofs.write("""\
from __future__ import absolute_import, division, print_function
import re
import gtk
class coot_serval_conf_list:
  def __init__(self):
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.set_title("Different conformations (Servalcat)")
    window.set_default_size(600, 600)
    scrolled_win = gtk.ScrolledWindow()
    scrolled_win.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
    vbox = gtk.VBox(False, 2)
    self.liststore = gtk.ListStore(str, int, str, int, str, str, float)
    self.filter = self.liststore.filter_new()
    self.treeview = gtk.TreeView(model=self.filter)
    for i, column_title in enumerate(["chain_1", "seq_1", "chain_2", "seq_2", "resn", "label", "diff"]):
      renderer = gtk.CellRendererText()
      column = gtk.TreeViewColumn(column_title, renderer, text=i)
      self.treeview.append_column(column)
    self.data = {}
    self.add_data()
    scrolled_win.add_with_viewport(self.treeview)  # add?
    vbox.pack_start(scrolled_win, True, True, 0)
    window.add(vbox)
    window.show_all()
    self.treeview.connect("row-activated", self.on_row_activated)

  def on_row_activated(self, treeview, path, column):
    assert len(path) == 1
    col_idx = [i for i, c in enumerate(treeview.get_columns()) if column == c][0]
    row = self.liststore[path[0]]
    if col_idx < 2:
      chain, resi = row[0], row[1]
    elif col_idx < 4:
      chain, resi = row[2], row[3]
    else:
      return
    if re.search("^[0-9]+_[0-9A-Za-z]", chain):
      chain = chain[chain.index("_")+1:]
    imol = active_atom_spec()[1][0]
    for name in (" CA ", " C1'"):
      a = get_atom(imol, chain, resi, "", name)
      if a:
        set_rotation_center(*a[2])
        break

  def add_data(self):
    for i, d in enumerate(self.data):
      self.liststore.append(d)

gui = coot_serval_conf_list()
""".format(for_coot))
    logger.writeln("\nRun:")
    logger.writeln(f"coot --script {coot_out}")
# compare_conf()

def adp_stats(args):
    if not args.output_prefix: args.output_prefix = fileio.splitext(os.path.basename(args.model))[0] + "_adp"
    st = fileio.read_structure(args.model)
    model.adp_analysis(st)
    b_all = [cra.atom.b_iso for cra in st[0].all() if cra.atom.occ > 0]
    
    # bin width from Freedman–Diaconis rule
    qs = numpy.quantile(b_all, [0, 0.25, 0.75, 1])
    bin_h = 2 * (qs[2] - qs[1]) / len(b_all)**(1/3.)
    
    # for plotly
    traces = []
    traces.append("x: [%s], type: 'histogram', name: 'All', xbins: {size: %f}"
                  % (",".join("%.2f"%x for x in b_all), bin_h))
    if len(st[0]) > 1:
        b_chain = {}
        for c in st[0]:
            b_chain.setdefault(c.name, []).extend(a.b_iso for r in c for a in r if a.occ > 0)
        for c in b_chain:
            bs = ",".join("%.2f" % x for x in b_chain[c])
            traces.append("x: [%s], type: 'histogram', name: 'Chain %s'" % (bs, c))
    with open(args.output_prefix + "_hist.html", "w") as ofs:
        ofs.write("""\
<html>
<head>
 <meta charset="utf-8" />
 <script src="https://cdn.plot.ly/plotly-2.20.0.min.js" charset="utf-8"></script>
</head>
<body>
 <div id="hist"></div>
 <script>
""")
        for i, t in enumerate(traces):
            ofs.write("var trace%d = {%s};\n" % (i+1, t))
        ofs.write("""\
  var layout = {
   title: "isotropic B histogram",
   xaxis: {title: "B"},
   yaxis: {title: "count"},
   barmode: "stack"
  };
  target = document.getElementById('hist');
  Plotly.newPlot(target, [%s], layout);
 </script>
</body>
</html>
""" % (",".join("trace%d" % (i+1) for i in range(len(traces)))))
        logger.writeln("check histogram: {}".format(ofs.name))
# adp_stats()

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
        mask = fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    hkldata = None
    labs = []
    for mapin in maps_in: # TODO rewrite in faster way
        ms = [fileio.read_ccp4_map(f) for f in mapin]
        d_min = args.resolution
        if d_min is None:
            d_min = maps.nyquist_resolution(ms[0][0])
            logger.writeln("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(d_min))
        tmp = maps.mask_and_fft_maps(ms, d_min, mask)
        labs.append("F{:02d}".format(len(labs)+1))
        tmp.df.rename(columns=dict(FP=labs[-1]), inplace=True)
        if hkldata is None:
            hkldata = tmp
        else:
            if hkldata.cell.parameters != tmp.cell.parameters: raise RuntimeError("Different unit cell!")
            hkldata.merge(tmp.df[["H","K","L",labs[-1]]])

    if not labs:
        raise SystemExit("No map files given. Exiting.")
            
    hkldata.setup_relion_binning("stat")

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
    abssqr = dict((lab, numpy.abs(hkldata.df[lab].to_numpy())**2) for lab in labs)
    for i_bin, idxes in hkldata.binned("stat"):
        bin_d_min = hkldata.binned_df["stat"].d_min[i_bin]
        bin_d_max = hkldata.binned_df["stat"].d_max[i_bin]
        ofs.write("{:.4f} {:7d} {:7.3f} {:7.3f}".format(1/bin_d_min**2, len(idxes), bin_d_max, bin_d_min,))
        for lab in labs:
            pwr = numpy.log10(numpy.average(abssqr[lab][idxes]))
            ofs.write(" {:.4e}".format(pwr))
        ofs.write("\n")
    ofs.write("$$\n")
    ofs.close()
# show_power()

def fcalc(args):
    if (args.auto_box_with_padding, args.cell).count(None) == 0:
        raise SystemExit("Error: you cannot specify both --auto_box_with_padding and --cell")
    
    if args.ligand: args.ligand = sum(args.ligand, [])
    if not args.output_prefix: args.output_prefix = "{}_fcalc_{}".format(fileio.splitext(os.path.basename(args.model))[0], args.source)

    st = fileio.read_structure(args.model)
    ccu = model.CustomCoefUtil()
    if not args.keep_charges:
        model.remove_charge([st])
    if args.source == "custom":
        ccu.read_from_cif(st, args.model)
        ccu.show_info()
        ccu.set_coeffs(st)
    else:
        model.check_atomsf([st], args.source)
    if not args.no_expand_ncs:
        model.expand_ncs(st)    

    if args.cell is not None:
        st.cell = gemmi.UnitCell(*args.cell)
    elif args.auto_box_with_padding is not None:
        st.cell = model.box_from_model(st[0], args.auto_box_with_padding)
        st.spacegroup_hm = "P 1"
        logger.writeln("Box size from the model with padding of {}: {}".format(args.auto_box_with_padding, st.cell.parameters))
        
    if not st.cell.is_crystal():
        raise SystemExit("ERROR: No unit cell information. Give --cell or --auto_box_with_padding.")

    if args.source=="electron" and st[0].has_hydrogen():
        monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                 stop_for_unknowns=False)
    else:
        monlib = None

    if args.method == "fft":
        fc_asu = model.calc_fc_fft(st, args.resolution, cutoff=args.cutoff, rate=args.rate,
                                   mott_bethe=args.source=="electron",
                                   monlib=monlib, source=args.source)
    else:
        fc_asu = model.calc_fc_direct(st, args.resolution, source=args.source,
                                      mott_bethe=args.source=="electron", monlib=monlib)

    hkldata = hkl.hkldata_from_asu_data(fc_asu, "FC")
    if args.as_intensity:
        hkldata.df["IC"] = numpy.abs(hkldata.df.FC)**2
        labout = ["IC"]
        if args.add_dummy_sigma:
            hkldata.df["SIGIC"] = 1.
            labout.append("SIGIC")
    else:
        labout = ["FC"]
        if args.add_dummy_sigma:
            hkldata.df["SIGFC"] = 1.
            labout.append("SIGFC")

    hkldata.write_mtz(args.output_prefix+".mtz", labout, types=dict(IC="J", SIGIC="Q", SIGFC="Q"))
# fcalc()

def nemap(args):
    from servalcat.spa import fofc

    if (args.trim or args.trim_mtz) and args.mask is None:
        raise SystemExit("\nError: You need to give --mask as you requested --trim or --trim_mtz.\n")
    
    if args.mask:
        mask = fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    halfmaps = fileio.read_halfmaps(args.halfmaps, pixel_size=args.pixel_size)
    if args.resolution is None:
        args.resolution = maps.nyquist_resolution(halfmaps[0][0])
        logger.writeln("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(args.resolution))

    d_min = args.resolution
    if args.local_fourier_weighting_with > 0:
        d_min = 1 / (args.local_fourier_weighting_with + 1 / d_min)
        logger.writeln("adjusting d_min= {:.2f} for local correlation".format(d_min))
    hkldata = maps.mask_and_fft_maps(halfmaps, d_min, mask)

    if args.local_fourier_weighting_with > 0:
        asu1 = hkldata.as_asu_data("F_map1")
        asu2 = hkldata.as_asu_data("F_map2")
        size = asu1.get_size_for_hkl(sample_rate=3)
        logger.writeln("using grid {}".format(size))
        gr1 = asu1.get_f_phi_on_grid(size)
        gr2 = asu2.get_f_phi_on_grid(size)
        kernel = ext.hard_sphere_kernel_recgrid(size, asu1.unit_cell, args.local_fourier_weighting_with)
        cc = maps.local_cc(gr1, gr2, kernel.array.real, method="simple")
        cc.array[cc.array < 0] = 0 # negative cc cannot be used anyway
        cc.array[:] = 2 * cc.array.real / (1 + cc.array.real) # to full map cc
        hkldata.df["cc"] = numpy.real(cc.get_value_by_hkl(hkldata.miller_array()))
        grf = type(gr1)((gr1.array + gr2.array) / 2, gr1.unit_cell, gr1.spacegroup)
        var_f = maps.local_var(grf, kernel.array.real, method="simple")
        hkldata.df["var_f"] = numpy.real(var_f.get_value_by_hkl(hkldata.miller_array()))
        if args.B is not None:
            k2_l = numpy.exp(-args.B / hkldata.d_spacings()**2 / 2)
            hkldata.df.cc = k2_l * hkldata.df.cc / (1 + (k2_l - 1) * hkldata.df.cc)
        hkldata.df["FWT"] = hkldata.df.FP * numpy.sqrt(hkldata.df.cc / hkldata.df.var_f)
        hkldata.df["kernel"] = numpy.real(kernel.get_value_by_hkl(hkldata.miller_array()))
        hkldata.write_mtz(args.output_prefix+"_cc.mtz", ["cc", "kernel"])
        hkldata = hkldata.copy(d_min=args.resolution)
        map_labs = ["FWT"]
    else:
        hkldata.setup_relion_binning("ml")
        maps.calc_noise_var_from_halfmaps(hkldata)
        map_labs = fofc.calc_maps(hkldata, B=args.B, has_halfmaps=True, half1_only=args.half1_only,
                                  no_fsc_weights=args.no_fsc_weights, sharpening_b=args.sharpening_b)
    fofc.write_files(hkldata, map_labs, grid_start=halfmaps[0][1], stats_str=None,
                     mask=mask, output_prefix=args.output_prefix,
                     trim_map=args.trim, trim_mtz=args.trim_mtz)
# nemap()

def blur(args):
    if args.output_prefix is None:
        args.output_prefix = fileio.splitext(os.path.basename(args.hklin))[0]
    
    if fileio.is_mmhkl_file(args.hklin):
        mtz = fileio.read_mmhkl(args.hklin)
        hkl.blur_mtz(mtz, args.B)
        suffix = ("_blur" if args.B > 0 else "_sharpen") + "_{:.2f}.mtz".format(abs(args.B))
        mtz.write_to_file(args.output_prefix+suffix)
        logger.writeln("Written: {}".format(args.output_prefix+suffix))
    else:
        raise SystemExit("ERROR: Unsupported file type: {}".format(args.hklin))
# blur()

def mask_from_model(args):
    st = fileio.read_structure(args.model) # TODO option to (or not to) expand NCS
    if args.selection:
        gemmi.Selection(args.selection).remove_not_selected(st)
    gr, grid_start, _ = fileio.read_ccp4_map(args.map, header_only=True)
    mask = maps.mask_from_model(st, args.radius, soft_edge=args.soft_edge, grid=gr)
    maps.write_ccp4_map(args.output, mask, grid_start=grid_start)
# mask_from_model()

def applymask(args):
    if args.output_prefix is None:
        args.output_prefix = fileio.splitext(os.path.basename(args.map))[0] + "_masked"

    grid, grid_start, _ = fileio.read_ccp4_map(args.map)
    mask = fileio.read_ccp4_map(args.mask)[0]
    logger.writeln("Applying mask")
    logger.writeln(" mask min: {:.3f} max: {:.3f}".format(numpy.min(mask), numpy.max(mask)))
    grid.array[:] *= mask.array

    if args.normalize:
        masked = grid.array[mask.array>args.mask_cutoff]
        masked_mean = numpy.average(masked)
        masked_std = numpy.std(masked)
        logger.writeln("Normalizing map values within mask")
        logger.writeln(" masked volume: {} mean: {:.3e} sd: {:.3e}".format(len(masked), masked_mean, masked_std))
        grid.array[:] = (grid.array - masked_mean) / masked_std

    maps.write_ccp4_map(args.output_prefix+".mrc", grid,
                        grid_start=grid_start,
                        mask_for_extent=mask.array if args.trim else None,
                        mask_threshold=args.mask_cutoff)
# applymask()

def map2mtz(args):
    if args.output is None:
        args.output = fileio.splitext(os.path.basename(args.map))[0] + "_fft.mtz"
    grid, grid_start, grid_shape = fileio.read_ccp4_map(args.map)
    if args.resolution is None:
        args.resolution = maps.nyquist_resolution(grid)
        logger.writeln("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(args.resolution))

    if grid_start != (0,0,0) or grid.shape != tuple(grid_shape):
        # If only subregion of whole grid in map, unit cell needs to be re-defined.
        if grid.shape != tuple(grid_shape):
            new_abc = [grid.unit_cell.parameters[i] * grid_shape[i] / grid.shape[i] for i in range(3)]
            cell = gemmi.UnitCell(*new_abc, *grid.unit_cell.parameters[3:])
            logger.writeln("Changing unit cell to {}".format(cell.parameters))
        else:
            cell = grid.unit_cell
        grid = gemmi.FloatGrid(grid.get_subarray(grid_start, grid_shape),
                               cell, grid.spacegroup)
    
    f_grid = gemmi.transform_map_to_f_phi(grid)
    asudata = f_grid.prepare_asu_data(dmin=args.resolution, with_000=True)
    hkldata = hkl.hkldata_from_asu_data(asudata, "F")
    if grid_start != (0,0,0):
        shifts = grid.get_position(*grid_start)
        hkldata.translate("F", shifts)
        logger.writeln("Applying phase shift with translation {}".format(shifts.tolist()))
    hkldata.write_mtz(args.output, ["F"])
# map2mtz()

def sm2mm(args):
    if args.output_prefix is None:
        args.output_prefix = os.path.basename(fileio.splitext(args.files[0])[0])
    st, mtz = fileio.read_small_molecule_files(args.files)
    if st is not None:
        fileio.write_model(st, prefix=args.output_prefix, pdb=True, cif=True)
    if mtz is not None:
        mtz_out = args.output_prefix + ".mtz"
        logger.writeln("Writing MTZ file: {}".format(mtz_out))
        mtz.write_to_file(mtz_out)
# sm2mm()

def mm2ins(args):
    if args.output is None:
        args.output = os.path.basename(fileio.splitext(args.files[0])[0]) + ".ins"
    st = fileio.read_structure(args.model)
    elems = [cra.atom.element.name for cra in st[0].all()]
    counts = {x:elems.count(x) for x in set(elems)}
    elems = sorted(counts)
    with open(args.output, "w") as ofs:
        ofs.write(f"SFAC {' '.join(elems)}\n")
        ofs.write(f"UNIT {' '.join(str(int(counts[x])) for x in elems)}\n")
        for cra in st[0].all():
            frac = st.cell.fractionalize(cra.atom.pos)
            u_iso = model.b_to_u * cra.atom.b_iso
            if cra.atom.is_hydrogen():
                u_iso = -1.2
            ofs.write(f"{cra.atom.name} {elems.index(cra.atom.element.name)+1} {frac.x:.6f} {frac.y:.6f} {frac.z:.6f} {10+cra.atom.occ} {u_iso:.5f}\n")
# mm2ins()

def seq(args):
    wrap_width = 100
    seqs = []
    if args.seq:
        args.seq = sum(args.seq, [])
        for sf in args.seq:
            seqs.extend(fileio.read_sequence_file(sf))
        
    sc = gemmi.AlignmentScoring()
    sc.match, sc.mismatch, sc.gapo, sc.gape, sc.good_gapo, sc.bad_gapo = args.scoring
    
    st = fileio.read_structure(args.model) # TODO option to (or not to) expand NCS
    model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)    
    for chain in st[0]:
        p = chain.get_polymer()
        if not p: continue
        p_type = p.check_polymer_type()
        if p_type in (gemmi.PolymerType.SaccharideD, gemmi.PolymerType.SaccharideL): continue
        p_seq = gemmi.one_letter_code(p.extract_sequence())
        results = []
        for name, seq in seqs:
            # what if DnaRnaHybrid?
            kind = {gemmi.PolymerType.Dna: gemmi.ResidueKind.DNA,
                    gemmi.PolymerType.Rna: gemmi.ResidueKind.RNA}.get(p_type, gemmi.ResidueKind.AA)
            s = [gemmi.expand_one_letter(x, kind) for x in seq]
            if None in s: continue
            #als = [gemmi.align_sequence_to_polymer(s, p, p_type, gemmi.AlignmentScoring(x)) for x in ("s", "p")]
            #results.append([name, max(als, key=lambda x: x.match_count), seq])
            results.append([name, gemmi.align_sequence_to_polymer(s, p, p_type, sc), seq])

        if results:
            logger.writeln("Chain: {}".format(chain.name))
            logger.writeln(" polymer type: {}".format(str(p_type).replace("PolymerType.", "")))
            name, al, s1 = max(results, key=lambda x: (x[1].match_count, x[1].score))
            logger.writeln(" match: {}".format(name))
            logger.writeln(" aligned: {}".format(al.match_count))
            logger.writeln(" score: {}".format(al.score))
            p1, p2 = al.add_gaps(s1, 1), al.add_gaps(p_seq, 2)
            unkseq = [x.start() for x in re.finditer(r"\-", p1)]
            mismatches = [x.start() for x in re.finditer(r"\.", al.match_string)]
            if mismatches or unkseq:
                idxes = {x.start(): i for i, x in enumerate(re.finditer("[^-]", p2))}
                seqnums = [str(x.seqid) for x in p]
                if mismatches:
                    logger.write(" mismatches: ")
                    logger.writeln(", ".join("{}({}>{})".format(seqnums[idxes[i]], p1[i], p2[i]) for i in mismatches))
                if unkseq:
                    logger.write(" unknown sequence: ")
                    logger.writeln(", ".join("{}({})".format(seqnums[idxes[i]], p2[i]) for i in unkseq))

            logger.writeln("")
            for i in range(0, len(p1), wrap_width):
                logger.writeln(" seq.  {}".format(p1[i:i+wrap_width]))
                logger.writeln("       {}".format(al.match_string[i:i+wrap_width]))
                logger.writeln(" model {}\n".format(p2[i:i+wrap_width]))
        else:
            logger.writeln("> Chain: {}".format(chain.name))
            logger.writeln(gemmi.one_letter_code(p.extract_sequence()))
            logger.writeln("")
# seq()

def dnarna(args):
    import scipy.spatial.transform
    rna_res = {"A":"DA", "G":"DG", "C":"DC", "U":"DT"}
    dna_res = {"DA":"A", "DG":"G", "DC":"C", "DT":"U"}
    if args.chains: args.chains = sum(args.chains, [])
    model_format = fileio.check_model_format(args.model)
    if not args.output:
        args.output = fileio.splitext(os.path.basename(args.model))[0] + "_conv" + model_format
    st = fileio.read_structure(args.model)
    if st[0].has_hydrogen():
        logger.writeln("Hydrogen atoms are detected. I cannot take care of them, so I will remove them.")
        st.remove_hydrogens()
    for chain in st[0]:
        if args.chains and chain.name not in args.chains:
            continue
        for res in chain:
            alt = "*" # XXX
            if res.name in rna_res and args.to_dna:
                logger.writeln(f"Changing {chain.name}/{res.seqid} {res.name} to DNA")
                res.name = rna_res[res.name]
                res.remove_atom("O2'", alt)
                if res.name == "DT":
                    C4 = res.find_atom("C4", alt)
                    C5 = res.find_atom("C5", alt)
                    C6 = res.find_atom("C6", alt)
                    v1 = C5.pos - C4.pos
                    v2 = C5.pos - C6.pos
                    v = v1 + v2
                    res.add_atom(C5)
                    res[-1].name = "C7"
                    res[-1].pos = C5.pos + v / v.length() * 1.5
            elif res.name in dna_res and args.to_rna:
                logger.writeln(f"Changing {chain.name}/{res.seqid} {res.name} to RNA")
                res.name = dna_res[res.name]
                C1p = numpy.array(res.find_atom("C1'", alt).pos.tolist())
                C2p = numpy.array(res.find_atom("C2'", alt).pos.tolist())
                C3p = numpy.array(res.find_atom("C3'", alt).pos.tolist())
                rotvec = C2p - C3p
                rotvec /= numpy.linalg.norm(rotvec)
                r = scipy.spatial.transform.Rotation.from_rotvec(-rotvec * 120,
                                                                 degrees=True)
                rotated = r.apply(C1p - C2p)
                rotated *= 1.411 / numpy.linalg.norm(rotated)
                res.add_atom(res.find_atom("O3'", alt))
                res[-1].name = "O2'"
                res[-1].pos.fromlist(C2p + rotated)
                if res.name == "U":
                    res.remove_atom("C7", alt)
    fileio.write_model(st, file_name=args.output)
# dnarna()

def show(args):
    for filename in args.files:
        ext = fileio.splitext(filename)[1]
        if ext in (".mrc", ".ccp4", ".map"):
            fileio.read_ccp4_map(filename)
            logger.writeln("\n")
# show()

def json2csv(args):
    if not args.output_prefix:
        args.output_prefix = fileio.splitext(os.path.basename(args.json))[0]
        
    df = pandas.read_json(args.json)
    df.to_csv(args.output_prefix+".csv", index=False)
    logger.writeln("Output: {}".format(args.output_prefix+".csv"))
# json2csv()

def main(args):
    comms = dict(show=show,
                 json2csv=json2csv,
                 symmodel=symmodel,
                 helical_biomt=helical_biomt,
                 expand=symexpand,
                 h_add=h_add,
                 add_op3=add_op3,
                 map_peaks=map_peaks,
                 h_density=h_density_analysis,
                 fix_link=fix_link,
                 merge_models=merge_models,
                 merge_dicts=merge_dicts,
                 geom=geometry,
                 conf=compare_conf,
                 adp=adp_stats,
                 power=show_power,
                 fcalc=fcalc,
                 nemap=nemap,
                 blur=blur,
                 mask_from_model=mask_from_model,
                 applymask=applymask,
                 map2mtz=map2mtz,
                 sm2mm=sm2mm,
                 mm2ins=mm2ins,
                 seq=seq,
                 dnarna=dnarna)
    
    com = args.subcommand
    f = comms.get(com)
    if f:
        return f(args)
    else:
        raise SystemExit("Unknown subcommand: {}".format(com))
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
