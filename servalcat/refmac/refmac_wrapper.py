"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
import json
import os
import sys
import tempfile
import subprocess
import argparse
from collections import OrderedDict
import servalcat # for version
from servalcat.utils import logger
from servalcat.refmac import refmac_keywords
from servalcat import utils

def add_arguments(parser):
    parser.description = 'Run REFMAC5 with gemmi-prepared restraints'
    parser.add_argument('--exe', default="refmac5", help='refmac5 binary')
    parser.add_argument("--monlib",
                        help="Monomer library path. Default: $CLIBD_MON")
    parser.add_argument('--ligand', nargs="*", action="append")
    parser.add_argument("opts", nargs="+",
                        help="HKLIN hklin XYZIN xyzin...")
    parser.add_argument('--auto_box_with_padding', type=float, help="Determine box size from model with specified padding")
    parser.add_argument('--no_adjust_hydrogen_distances', action='store_true', help="By default it adjusts hydrogen distances using ideal values. This option is to disable it.")
    parser.add_argument('--keep_original_output', action='store_true', help="with .org extension")
    parser.add_argument("--keep_entities", action='store_true',
                        help="Do not override entities")
    parser.add_argument("--tls_addu", action='store_true',
                        help="Write prefix_addu.mmcif where TLS contribution is added to aniso U. Don't use this with 'tlso addu' keyword.")
    parser.add_argument('--prefix', help="output prefix")
    parser.add_argument("-v", "--version", action="version",
                        version=logger.versions_str())
    # TODO --cell to override unit cell?

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def read_stdin(stdin):
    print("Waiting for input..")
    # these make keywords will be ignored (just passed to refmac): ribo,valu,spec,form,sdmi,segi
    ret = {"make":{}, "ridge":{}, "refi":{}}
    inputs = []
    for l in refmac_keywords.get_lines(stdin):
        refmac_keywords.parse_line(l, ret)
        inputs.append(l + "\n")
    
    def sorry(s): raise SystemExit("Sorry, '{}' is not supported".format(s))
    if ret["make"].get("hydr") == "f":
        sorry("make hydr full")
    if ret["make"].get("buil") == "y":
        sorry("make build yes")
    return inputs, ret
# read_stdin()

def prepare_crd(st, crdout, ligand, make, monlib_path=None, h_pos="elec",
                no_adjust_hydrogen_distances=False, fix_long_resnames=True,
                keep_entities=False, unre=False):
    assert h_pos in ("elec", "nucl")
    h_change = dict(a=gemmi.HydrogenChange.ReAddButWater,
                    y=gemmi.HydrogenChange.NoChange,
                    n=gemmi.HydrogenChange.Remove)[make.get("hydr", "a")]
    utils.model.fix_deuterium_residues(st)
    for chain in st[0]:
        if not chain.name:
            chain.name = "X" # Refmac behavior. Empty chain name will cause a problem
        for res in chain:
            if res.is_water():
                res.name = "HOH"
            for at in res:
                if at.occ > 1: # XXX should I check special positions?
                    at.occ = 1.

    refmac_fixes = utils.refmac.FixForRefmac()
    if keep_entities:
        refmac_fixes.store_res_labels(st)
    else:
        utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)

    if unre:
        logger.writeln("Monomer library will not be loaded due to unrestrained refinement request")
        monlib = gemmi.MonLib()
    else:
        # TODO read dictionary from xyzin (priority: user cif -> monlib -> xyzin
        try:
            monlib = utils.restraints.load_monomer_library(st,
                                                           monomer_dir=monlib_path,
                                                           cif_files=ligand,
                                                           stop_for_unknowns=not make.get("newligand"))
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        use_cispeps = make.get("cispept", "y") != "y"
        make_link = make.get("link", "n")
        make_ss = make.get("ss", "y")
        only_from = set()
        if make_link == "y":
            # add all links
            add_found = True
        elif make_ss == "y":
            add_found = True
            only_from.add("disulf")
        else:
            add_found = False

        utils.restraints.fix_elements_in_model(monlib, st)
        utils.restraints.find_and_fix_links(st, monlib, add_found=add_found,
                                            find_metal_links=(make_link == "y"),
                                            find_symmetry_related=False, add_only_from=only_from)
        for con in st.connections:
            if con.link_id not in ("?", "", "gap") and con.link_id not in monlib.links:
                logger.writeln(" removing unknown link id ({}). Ad-hoc link will be generated.".format(con.link_id))
                con.link_id = ""

    max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
    if max_seq_num > 9999:
        logger.writeln("Max residue number ({}) exceeds 9999. Needs workaround.".format(max_seq_num))
        topo = gemmi.prepare_topology(st, monlib, warnings=logger.silent(), ignore_unknown_links=True)
        refmac_fixes.fix_before_topology(st, topo, 
                                         fix_microheterogeneity=False,
                                         fix_resimax=True,
                                         fix_nonpolymer=False)

    if unre:
        # Refmac5 does not seem to do anything to hydrogen when unre regardless of "make hydr"
        topo = gemmi.prepare_topology(st, monlib, warnings=logger.silent(), ignore_unknown_links=True)
        metal_kws = []
    else:
        if make.get("hydr") == "a": logger.writeln("(re)generating hydrogen atoms")
        try:
            topo, metal_kws = utils.restraints.prepare_topology(st, monlib, h_change=h_change, ignore_unknown_links=False,
                                                                check_hydrogen=(h_change==gemmi.HydrogenChange.NoChange),
                                                                use_cispeps=use_cispeps)
        except RuntimeError as e:
            raise SystemExit("Error: {}".format(e))

        if make.get("hydr") != "n" and st[0].has_hydrogen():
            if h_pos == "nucl" and (make.get("hydr") == "a" or not no_adjust_hydrogen_distances):
                resnames = st[0].get_all_residue_names()
                utils.restraints.check_monlib_support_nucleus_distances(monlib, resnames)
                logger.writeln("adjusting hydrogen position to nucleus")
                topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=1.1)
            elif h_pos == "elec" and make.get("hydr") == "y" and not no_adjust_hydrogen_distances:
                logger.writeln("adjusting hydrogen position to electron cloud")
                topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)

    if fix_long_resnames: refmac_fixes.fix_long_resnames(st)

    # remove "given" ncs matrices
    # TODO write them back to the output files
    st.ncs = gemmi.NcsOpList(x for x in st.ncs if not x.given)

    # for safety
    if "_entry.id" in st.info:
        st.info["_entry.id"] = st.info["_entry.id"].replace(" ", "")
    date_key = "_pdbx_database_status.recvd_initial_deposition_date"
    if date_key in st.info:
        tmp = st.info[date_key]
        if len(tmp) > 5 and tmp[4] == "-":
            if len(tmp) > 8 and tmp[8] != "" and not tmp[5:7].isdigit():
                tmp = "XX"
        elif len(tmp) > 6 and tmp[5] == "-":
            if not tmp[3:5].isdigit():
                tmp = "XX"
        st.info[date_key] = tmp
    # internal chain ID
    for chain in st[0]:
        for res in chain:
            if len(chain.name) < 3:
                # Change Axp (or AAxp) to A_p (or AA_p)
                res.subchain = chain.name + "_" + res.subchain[len(chain.name)+1:]
            else:
                # Refmac only expects '_' at 2nd or 3rd position, and can accept up to 4 letters.
                # Using raw chain ID may change alignment result for local NCS restraints,
                # but to avoid this we would need chain ID translation, which is too complicated.
                # This also invalidates _struct_asym, which Refmac does not seem to care
                res.subchain = chain.name
    # change st.name if needed
    block_names = utils.restraints.dictionary_block_names(monlib, topo)
    for i in range(1000):
        if st.name.lower() in block_names:
            st.name = st.name + str(i)
    doc = gemmi.prepare_refmac_crd(st, topo, monlib, h_change)
    doc.write_file(crdout, options=gemmi.cif.Style.NoBlankLines)
    logger.writeln("crd file written: {}".format(crdout))
    return refmac_fixes, [x+"\n" for x in metal_kws]
# prepare_crd()

def get_output_model_names(xyzout):
    # ref: WRITE_ATOMS_REFMAC in oppro_allocate.f
    if xyzout is None: xyzout = "XYZOUT"
    pdb, mmcif = "", ""
    if len(xyzout) > 3:
        if xyzout.lower().endswith("pdb"):
            mmcif = xyzout[:-4] + ".mmcif"
            pdb = xyzout
        else:
            if xyzout.lower().endswith("cif") and len(xyzout) > 5:
                if xyzout.lower().endswith("mmcif"):
                    mmcif = xyzout
                    pdb = xyzout[:-6] + ".pdb"
                else:
                    mmcif = xyzout
                    pdb = xyzout[:-4] + ".pdb"
            else:
                mmcif = xyzout + ".mmcif"
                pdb = xyzout
    else:
        mmcif = xyzout + ".mmcif"
        pdb = xyzout
        
    return pdb, mmcif
# get_output_model_names()

def modify_output(pdbout, cifout, fixes, hout, cispeps, software_items, keep_original_output=False, tls_addu=False):
    st = utils.fileio.read_structure(cifout)
    st.cispeps = cispeps
    if os.path.exists(pdbout):
        st_pdb = gemmi.read_pdb(pdbout)
        st.raw_remarks = st_pdb.raw_remarks
        # Refmac only writes NCS matrices to pdb
        st.ncs = st_pdb.ncs
        if "_struct_ncs_oper.id" in st_pdb.info: # to write identity operator
            st.info["_struct_ncs_oper.id"] = st_pdb.info["_struct_ncs_oper.id"]
    if fixes is not None:
        fixes.modify_back(st)
    for con in st.connections:
        if con.link_id == "disulf":
            con.type = gemmi.ConnectionType.Disulf
        # should we check metals and put MetalC?

    # fix entity (Refmac seems to make DNA non-polymer; as seen in 1fix)
    if not fixes or not fixes.res_labels:
        utils.model.setup_entities(st, clear=True, overwrite_entity_type=True, force_subchain_names=True)
        for e in st.entities:
            if not e.full_sequence and e.entity_type == gemmi.EntityType.Polymer and e.subchains:
                rspan = st[0].get_subchain(e.subchains[0])
                e.full_sequence = [r.name for r in rspan]

        # fix label_seq_id
        for chain in st[0]:
            for res in chain:
                res.label_seq = None
        st.assign_label_seq_id()

    # add servalcat version
    if len(st.meta.software) > 0 and st.meta.software[-1].name == "refmac":
        st.meta.software[-1].version += f" (refmacat {servalcat.__version__})"
    st.meta.software = software_items + st.meta.software
    
    suffix = ".org"
    os.rename(cifout, cifout + suffix)
    utils.fileio.write_mmcif(st, cifout, cifout + suffix)

    if tls_addu:
        doc_ref = gemmi.cif.read(cifout + suffix)
        tls_groups = {int(x.id): x for x in st.meta.refinement[0].tls_groups}
        tls_details = doc_ref[0].find_value("_ccp4_refine_tls.details")
        if tls_groups and gemmi.cif.as_string(tls_details) == "U values: residual only":
            st2 = st.clone()
            for cra in st2[0].all():
                tlsgr = tls_groups.get(cra.atom.tls_group_id)
                if cra.atom.tls_group_id > 0 and tlsgr is not None:
                    if not cra.atom.aniso.nonzero():
                        u = cra.atom.b_iso * utils.model.b_to_u
                        cra.atom.aniso = gemmi.SMat33f(u, u, u, 0, 0, 0)
                    u_from_tls = gemmi.calculate_u_from_tls(tlsgr, cra.atom.pos)
                    cra.atom.aniso += gemmi.SMat33f(*u_from_tls.elements_pdb())
                    cra.atom.b_iso = cra.atom.aniso.trace() / 3. * utils.model.u_to_b
            cifout2 = cifout[:cifout.rindex(".")] + "_addu" + cifout[cifout.rindex("."):]
            doc_ref[0].set_pair("_ccp4_refine_tls.details", gemmi.cif.quote("U values: with tls added"))
            utils.fileio.write_mmcif(st2, cifout2, cif_ref_doc=doc_ref)
        else:
            if not tls_groups:
                msg = "TLS group definition not found in the model"
            else:
                msg = "TLS already applied to the U values"
            logger.writeln(f"Error: --tls_addu requested, but {msg}.")

    if st.has_d_fraction:
        st.store_deuterium_as_fraction(False) # also useful for pdb
        logger.writeln("will write a H/D expanded mmcif file")
        cifout2 = cifout[:cifout.rindex(".")] + "_hd_expand" + cifout[cifout.rindex("."):]
        utils.fileio.write_mmcif(st, cifout2, cifout + suffix)
    
    chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
    seqnums = [res.seqid.num for chain in st[0] for res in chain]
    if chain_id_len_max > 1 or min(seqnums) <= -1000 or max(seqnums) >= 10000:
        logger.writeln("This structure cannot be saved as an official PDB format. Using hybrid-36. Header part may be inaccurate.")
    if not hout:
        st.remove_hydrogens() # remove hydrogen from pdb, while kept in mmcif
    os.rename(pdbout, pdbout + suffix)
    utils.fileio.write_pdb(st, pdbout)
    if not keep_original_output:
        os.remove(pdbout + suffix)
        os.remove(cifout + suffix)
# modify_output()

def main(args):
    if len(args.opts) % 2 != 0: raise SystemExit("Invalid number of args")
    args.ligand = sum(args.ligand, []) if args.ligand else []

    inputs, keywords = read_stdin(sys.stdin) # TODO read psrestin also?
    if not keywords["make"].get("exit"):
        refmac_ver = utils.refmac.check_version(args.exe)
        if not refmac_ver:
            raise SystemExit("Error: Check Refmac installation or use --exe to give the location.")
        if refmac_ver < (5, 8, 404):
            raise SystemExit("Error: this version of Refmac is not supported. Update to 5.8.404 or newer")

    opts = OrderedDict((args.opts[2*i].lower(), args.opts[2*i+1]) for i in range(len(args.opts)//2))
    xyzin = opts.get("xyzin")
    xyzout = opts.get("xyzout")
    libin = opts.pop("libin", None)
    if libin: args.ligand.append(libin)
    if not args.monlib:
        # if --monlib is given, it has priority.
        args.monlib = opts.pop("clibd_mon", None)
    for k in ("temp1", "scrref"): # scrref has priority
        if k in opts:
            logger.writeln("updating CCP4_SCR from {}={}".format(k, opts[k]))
            os.environ["CCP4_SCR"] = os.path.dirname(opts[k]) # XXX "." may be given, which causes problem (os.path.isdir("") is False)
    utils.refmac.ensure_ccp4scr()
    if args.prefix:
        if "xyzin" in opts and "xyzout" not in opts: opts["xyzout"] = args.prefix + ".pdb"
        if "hklin" in opts and "hklout" not in opts: opts["hklout"] = args.prefix + ".mtz"
        if "tlsin" in opts and "tlsout" not in opts: opts["tlsout"] = args.prefix + ".tls"
        
    # TODO what if restin is given or make cr prepared is given?
    # TODO check make pept/link/suga/ss/conn/symm/chain

    if "hklin" in opts: # for history
        software_items = utils.fileio.software_items_from_mtz(opts["hklin"])
    else:
        software_items = []
    
    # Process model
    crdout = None
    refmac_fixes = None
    cispeps = []
    unre = keywords["refi"].get("type") == "unre"
    if xyzin is not None:
        #tmpfd, crdout = tempfile.mkstemp(prefix="gemmi_", suffix=".crd") # TODO use dir=CCP4_SCR
        #os.close(tmpfd)
        st = utils.fileio.read_structure(xyzin)
        if not st.cell.is_crystal() and not unre:
            if args.auto_box_with_padding is not None:
                st.cell = utils.model.box_from_model(st[0], args.auto_box_with_padding)
                st.spacegroup_hm = "P 1"
                logger.writeln("Box size from the model with padding of {}: {}".format(args.auto_box_with_padding, st.cell.parameters))
            else:
                raise SystemExit("Error: unit cell is not defined in the model.")
        xyzout_dir = os.path.dirname(get_output_model_names(opts.get("xyzout"))[0])
        crdout = os.path.join(xyzout_dir,
                              "gemmi_{}_{}.crd".format(utils.fileio.splitext(os.path.basename(xyzin))[0], os.getpid()))
        refmac_fixes, metal_kws = prepare_crd(st, crdout, args.ligand, make=keywords["make"], monlib_path=args.monlib,
                                              h_pos="nucl" if keywords.get("source")=="ne" else "elec",
                                              no_adjust_hydrogen_distances=args.no_adjust_hydrogen_distances,
                                              keep_entities=args.keep_entities,
                                              unre=unre)
        inputs = metal_kws + inputs # add metal exte first; otherwise it may be affected by user-defined inputs
        opts["xyzin"] = crdout
        cispeps = st.cispeps

    if keywords["make"].get("exit"):
        return

    # Run Refmac
    cmd = [args.exe] + list(sum(tuple(opts.items()), ()))
    env = os.environ
    logger.writeln("Running REFMAC5..")
    if args.monlib:
        logger.writeln("CLIBD_MON={}".format(args.monlib))
        env["CLIBD_MON"] = os.path.join(args.monlib, "") # should end with /
    logger.writeln(" ".join(cmd))
    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, env=env)
    if crdout: p.stdin.write("make cr prepared\n")
    p.stdin.write("".join(inputs))
    p.stdin.close()
    # prepare conversion for long residue names
    resn_conv = {}
    if refmac_fixes:
        for old, new in refmac_fixes.resn_old_new:
            n = "{:4s}".format(old)
            if len(n) > 4: n += " "
            resn_conv[new] = n
    # print raw output
    for l in iter(p.stdout.readline, ""):
        for tn in resn_conv:
            l = l.replace(tn, resn_conv[tn])
        logger.write(l)
    retcode = p.wait()
    logger.writeln("\nRefmac finished with exit code= {}".format(retcode))

    if not args.keep_original_output and crdout and os.path.exists(crdout):
        os.remove(crdout)

    # Modify output
    if xyzin is not None:
        pdbout, cifout = get_output_model_names(opts.get("xyzout"))
        if os.path.exists(cifout):
            modify_output(pdbout, cifout, refmac_fixes, keywords["make"].get("hout"), cispeps,
                          software_items, args.keep_original_output, args.tls_addu)
# main()

def command_line():
    import sys
    args = parse_args(sys.argv[1:])
    if args.prefix:
        logger.set_file(args.prefix + ".log")
    logger.write_header(command="refmacat")
    main(args)

if __name__ == "__main__":
    command_line()
