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
import os
import gemmi
import numpy
import scipy.spatial
import pandas
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
    group.add_argument('--cell', type=float, nargs=6, help="Box size")
    sym_group = parser.add_argument_group("symmetry")
    symmetry.add_symmetry_args(sym_group, require_pg=True)
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
    group.add_argument('--cell', type=float, nargs=6, help="Box size")
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
    parser.add_argument('--bond_margin', type=float, default=1.1, help='(default: %(default).1f)')
    parser.add_argument('--remove_unknown', action="store_true")
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
    parser.add_argument('--sigma', type=float, default=5,
                        help="sigma cutoff to print outliers (default: %(default).1f)")
    parser.add_argument('--write_z_per_atom', nargs="*", 
                        help="write model file(s) with sum of z values of specified metric as B values")
    parser.add_argument('-o', '--output_prefix', default="geometry",
                        help="default: %(default)s")

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
    parser.add_argument("--source", choices=["electron", "xray", "neutron"], default="electron")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--cell', type=float, nargs=6, help="Override unit cell")
    parser.add_argument('--auto_box_with_padding', type=float, help="Determine box size from model with specified padding")
    parser.add_argument('--cutoff', type=float, default=1e-7)
    parser.add_argument('--rate', type=float, default=1.5)
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
    parser.add_argument("-d", '--resolution', type=float, required=True)
    parser.add_argument('-m', '--mask', help="mask file")
    parser.add_argument('-o', '--output_prefix', default='nemap')
    parser.add_argument("--trim", action='store_true', help="Write trimmed maps")
    parser.add_argument("--trim_mtz", action='store_true', help="Write trimmed mtz")

    # blur
    parser = subparsers.add_parser("blur", description = 'Blur data by specified B value')
    parser.add_argument('--hklin', required=True, help="input MTZ file")
    parser.add_argument('-B', type=float, required=True, help="B value for blurring (negative value for sharpening)")
    parser.add_argument('-o', '--output_prefix')

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
        raise SystemExit("ERROR: give both helical paramters --twist and --rise")

    is_helical = args.twist is not None
    st, cif_ref = fileio.read_structure_from_pdb_and_mmcif(args.model)
    st.spacegroup_hm = "P 1"
    map_and_start = None
    if args.map:
        logger.write("Reading cell from map")
        map_and_start = fileio.read_ccp4_map(args.map)
        st.cell = map_and_start[0].unit_cell
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        raise SystemExit("Error: Unit cell parameters look wrong. Please use --map or --cell")

    if args.chains:
        logger.write("Keep {} chains only".format(" ".join(args.chains)))
        chains = set(args.chains)
        for m in st:
            to_del = [c.name for c in m if c.name not in chains]
            for c in to_del: m.remove_chain(c)
        if st[0].count_atom_sites() == 0:
            raise SystemExit("ERROR: no atoms left. Check --chains option.")

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    symmetry.update_ncs_from_args(args, st, map_and_start=map_and_start)

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
        raise SystemExit("ERROR: give helical paramters --twist and --rise")

    model_format = fileio.check_model_format(args.model)
    howtoname = dict(dup=gemmi.HowToNameCopiedChain.Dup,
                     short=gemmi.HowToNameCopiedChain.Short,
                     number=gemmi.HowToNameCopiedChain.AddNumber)[args.howtoname]

    st, cif_ref = fileio.read_structure_from_pdb_and_mmcif(args.model)
    st.spacegroup_hm = "P 1"
    map_and_start = None
    if args.map:
        logger.write("Reading cell from map")
        map_and_start = fileio.read_ccp4_map(args.map)
        st.cell = map_and_start[0].unit_cell
    elif args.cell:
        st.cell = gemmi.UnitCell(*args.cell)
    elif not st.cell.is_crystal():
        raise SystemExit("Error: Unit cell parameters look wrong. Please use --map or --cell")

    all_chains = [c.name for c in st[0] if c.name not in st[0]]

    ncsops = symmetry.ncsops_from_args(args, st.cell, map_and_start=map_and_start, st=st,
                                       helical_min_n=args.start, helical_max_n=args.end)
    #ncsops = [x for x in ncsops if not x.tr.is_identity()] # remove identity

    logger.write("")
    logger.write("-------------------------------------------------------------")
    logger.write("You may need to write following matrices in OneDep interface:")
    for idx, op in enumerate(ncsops):
        logger.write("")
        logger.write("operator {}".format(idx+1))
        mat = op.tr.mat.tolist()
        vec = op.tr.vec.tolist()
        for i in range(3):
            mstr = ["{:10.6f}".format(mat[i][j]) for j in range(3)]
            logger.write("{} {:14.5f}".format(" ".join(mstr), vec[i]))
    logger.write("-------------------------------------------------------------")
    logger.write("")

    # BIOMT
    st.assemblies.clear()
    st.raw_remarks = []
    a = model.prepare_assembly("1", all_chains, ncsops, is_helical=True)
    st.assemblies.append(a)

    if not args.output_prfix:
        args.output_prfix = fileio.splitext(os.path.basename(args.model))[0] + "_biomt"

    fileio.write_model(st, args.output_prfix, pdb=(model_format == ".pdb"), cif=True, cif_ref=cif_ref)
    logger.write("")
    logger.write("These {}.* files may be used for deposition (once OneDep implemented reading BIOMT from file..)".format(args.output_prfix))
    logger.write("")
    # BIOMT expand
    st.transform_to_assembly("1", howtoname)
    args.output_prfix += "_expanded"
    fileio.write_model(st, file_name=args.output_prfix+model_format)
    logger.write(" note that this expanded model file is just for visual inspection, *not* for deposition!")
# helical_biomt()

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

def h_density_analysis(args):
    #if args.source != "electron":
    #    raise SystemExit("Only electron source is supported.")
    model_format = fileio.check_model_format(args.model)
    st = fileio.read_structure(args.model)
    if st[0].count_hydrogen_sites() == 0:
        raise SystemExit("No hydrogen in model.")

    if args.output_prefix is None:
        args.output_prefix = fileio.splitext(os.path.basename(args.model))[0] + "_hana"

    if args.mtz is not None:
        mtz = gemmi.read_mtz_file(args.mtz)
        lab_f, lab_phi = args.mtz_labels.split(",")
        asu = mtz.get_f_phi(lab_f, lab_phi)
        if args.oversample_pixel is not None:
            d_min = numpy.min(asu.make_d_array())
            sample_rate = d_min / args.oversample_pixel
        else:
            sample_rate = 3
        gr = asu.transform_f_phi_to_map(sample_rate=sample_rate)
    elif args.map is not None:
        gr = fileio.read_ccp4_map(args.map)[0]
        if args.oversample_pixel is not None:
            asu = gemmi.transform_map_to_f_phi(gr).prepare_asu_data()
            d_min = numpy.min(asu.make_d_array())
            sample_rate = d_min / args.oversample_pixel
            gr = asu.transform_f_phi_to_map(sample_rate=sample_rate)
    else:
        raise SystemExit("Invalid input")

    if args.oversample_pixel is not None:
        logger.write("--oversample_pixel= {} is requested.".format(args.oversample_pixel))
        logger.write(" recalculated grid:")
        logger.write("  {:4d} {:4d} {:4d}".format(*gr.shape))
        logger.write(" spacings:")
        logger.write("  {:.6f} {:.6f} {:.6f}".format(*gr.spacing))
        #maps.write_ccp4_map("{}_oversampled.mrc".format(args.output_prefix), gr)

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
    logger.write("")
    logger.write("Found hydrogen peaks:")
    logger.write("dist map  vol  atom")
    for _, map_val, dist, volume, chain, resi, resn, atom, alt in found:
        logger.write("{:.2f} {:.2f} {:.2f} {}/{} {}/{}{}".format(dist, map_val, volume,
                                                                 chain, resn, resi,
                                                                 atom, "."+alt if alt else ""))

    logger.write("")
    logger.write("Result:")
    logger.write(" number of hydrogen in the model  : {}".format(n_hydr))
    logger.write(" number of peaks close to hydrogen: {} ({:.1%})".format(len(found), len(found)/n_hydr))
    logger.write("")

    st_peaks = model.st_from_positions([getpos(b) for b in blobs],
                                       bs=[gr.interpolate_value(getpos(b)) for b in blobs],
                                       qs=h_assigned)
    fileio.write_model(st_peaks, file_name="{}_peaks.mmcif".format(args.output_prefix))
    logger.write(" this file includes peak positions")
    logger.write(" occ=1: hydrogen assigned, occ=0: unassigned.")
    logger.write(" B: density value at {}".format(args.blob_pos))
    logger.write("")
    
    fileio.write_model(st2, file_name="{}_h_with_peak{}".format(args.output_prefix, model_format))
    logger.write(" this file is a copy of input model, where hydrogen atoms without peaks are removed.")
# h_density_analysis()

def fix_link(args):
    st = fileio.read_structure(args.model)
    model_format = fileio.check_model_format(args.model)
    
    if not args.output:
        tmp = fileio.splitext(os.path.basename(args.model))[0]
        args.output = tmp + "_fixlink" + model_format
    logger.write("Output file: {}".format(args.output))
        
    args.ligand = sum(args.ligand, []) if args.ligand else []
    monlib = restraints.load_monomer_library(st,
                                             monomer_dir=args.monlib,
                                             cif_files=args.ligand)
    restraints.find_and_fix_links(st, monlib, bond_margin=args.bond_margin,
                                  remove_unknown=args.remove_unknown)
    fileio.write_model(st, file_name=args.output)
# fix_link()
    
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

def merge_dicts(args):
    fileio.merge_ligand_cif(args.cifs, args.output)
# merge_dicts()

def geometry(args):
    if args.ligand: args.ligand = sum(args.ligand, [])
    args.write_z_per_atom = set(args.write_z_per_atom) if args.write_z_per_atom else set()
    if not args.write_z_per_atom.issubset(set(("bond","angle","chiral","plane"))):
        raise SystemExit("invalid keyword included in --write_z_per_atom: {}".format(args.write_z_per_atom))

    model_format = fileio.check_model_format(args.model)
    st = fileio.read_structure(args.model)
    try:
        monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                                 stop_for_unknowns=True, check_hydrogen=True)
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))
    
    restr = restraints.Restraints(st, monlib)
    for k in restr.outlier_sigmas: restr.outlier_sigmas[k] = args.sigma
    dfs = restr.show_all()
    logger.write("")
    for k in dfs:
        json_out = "{}_{}.json".format(args.output_prefix, k)
        with open(json_out, "w") as ofs: dfs[k].to_json(ofs, indent=2, orient="index")
        logger.write("written: {}".format(json_out))

    for k in args.write_z_per_atom:
        for cra in st[0].all(): cra.atom.b_iso = 0
        if k == "bond":
            for t in restr.topo.bonds:
                z = t.calculate_z()
                for a in t.atoms: a.b_iso += z
        elif k == "angle":
            for t in restr.topo.angles:
                t.atoms[1].b_iso += t.calculate_z()
        elif k == "chiral":
            for t in restr.topo.chirs:
                t.atoms[0].b_iso += t.calculate_z(restr.topo.ideal_chiral_abs_volume(t), 0.2)
        elif k == "plane":
            for t in restr.topo.planes:
                devs = restraints.plane_deviations(t.atoms)
                for a, d in zip(t.atoms, devs):
                    a.b_iso += abs(d) / t.restr.esd
  
        fileio.write_model(st, file_name="z_{}s{}".format(k, model_format))

# geometry()

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
            logger.write("WARNING: --resolution is not specified. Using Nyquist resolution: {:.2f}".format(d_min))
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
            
    hkldata.setup_relion_binning()

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
    for i_bin, idxes in hkldata.binned():
        bin_d_min = hkldata.binned_df.d_min[i_bin]
        bin_d_max = hkldata.binned_df.d_max[i_bin]
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
    if not args.no_expand_ncs:
        model.expand_ncs(st)    

    if args.cell is not None:
        st.cell = gemmi.UnitCell(*args.cell)
    elif args.auto_box_with_padding is not None:
        st.cell = model.box_from_model(st[0], args.auto_box_with_padding)
        st.spacegroup_hm = "P 1"
        logger.write("Box size from the model with padding of {}: {}".format(args.auto_box_with_padding, st.cell.parameters))
        
    if not st.cell.is_crystal():
        raise SystemExit("ERROR: No unit cell information. Give --cell or --auto_box_with_padding.")

    monlib = restraints.load_monomer_library(st, monomer_dir=args.monlib, cif_files=args.ligand, 
                                             stop_for_unknowns=False, check_hydrogen=True)

    if args.method == "fft":
        fc_asu = model.calc_fc_fft(st, args.resolution, cutoff=args.cutoff, rate=args.rate,
                                   mott_bethe=args.source=="electron",
                                   monlib=monlib, source=args.source)
    else:
        fc_asu = model.calc_fc_direct(st, args.resolution, source=args.source,
                                      mott_bethe=args.source=="electron", monlib=monlib)

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

def nemap(args):
    from servalcat.spa import fofc

    if (args.trim or args.trim_mtz) and args.mask is None:
        raise SystemExit("\nError: You need to give --mask as you requested --trim or --trim_mtz.\n")
    
    if args.mask:
        mask = fileio.read_ccp4_map(args.mask)[0]
    else:
        mask = None

    halfmaps = [fileio.read_ccp4_map(f, pixel_size=args.pixel_size) for f in args.halfmaps]
    hkldata = maps.mask_and_fft_maps(halfmaps, args.resolution, mask)
    hkldata.setup_relion_binning()
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
    
    if args.hklin.endswith(".mtz"):
        mtz = gemmi.read_mtz_file(args.hklin)
        hkl.blur_mtz(mtz, args.B)
        suffix = ("_blur" if args.B > 0 else "_sharpen") + "_{:.2f}.mtz".format(abs(args.B))
        mtz.write_to_file(args.output_prefix+suffix)
        logger.write("Written: {}".format(args.output_prefix+suffix))
    else:
        raise SystemExit("ERROR: Unsupported file type: {}".format(args.hklin))
# blur()

def applymask(args):
    if args.output_prefix is None:
        args.output_prefix = fileio.splitext(os.path.basename(args.map))[0] + "_masked"

    grid, grid_start = fileio.read_ccp4_map(args.map)
    mask = fileio.read_ccp4_map(args.mask)[0]
    logger.write("Applying mask")
    logger.write(" mask min: {:.3f} max: {:.3f}".format(numpy.min(mask), numpy.max(mask)))
    grid.array[:] *= mask.array

    if args.normalize:
        masked = grid.array[mask.array>args.mask_cutoff]
        masked_mean = numpy.average(masked)
        masked_std = numpy.std(masked)
        logger.write("Normalizing map values within mask")
        logger.write(" masked volume: {} mean: {:.3e} sd: {:.3e}".format(len(masked), masked_mean, masked_std))
        grid.array[:] = (grid.array - masked_mean) / masked_std

    maps.write_ccp4_map(args.output_prefix+".mrc", grid,
                        grid_start=grid_start,
                        mask_for_extent=mask.array if args.trim else None,
                        mask_threshold=args.mask_cutoff)
# applymask()
    
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
                 helical_biomt=helical_biomt,
                 expand=symexpand,
                 h_add=h_add,
                 h_density=h_density_analysis,
                 fix_link=fix_link,
                 merge_models=merge_models,
                 merge_dicts=merge_dicts,
                 geom=geometry,
                 power=show_power,
                 fcalc=fcalc,
                 nemap=nemap,
                 blur=blur,
                 applymask=applymask)
    
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
