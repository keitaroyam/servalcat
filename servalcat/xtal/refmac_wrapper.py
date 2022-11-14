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
from servalcat.utils import logger
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

    # TODO --prefix to automatically set hklout/xyzout/log file?
    # TODO --cell to override unit cell

# add_arguments()
                        
def parse_args(arg_list):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(arg_list)
# parse_args()

def get_opt(opts, kwd, keep=False):
    for i in range(len(opts)-1):
        if opts[i].lower() == kwd.lower():
            return opts[i+1], opts if keep else (opts[:i] + opts[i+2:])
    return None, opts
# get_opt()

def parse_keywords(inputs):
    # these make keywords will be ignored (just passed to refmac): hout,ribo,valu,spec,form,sdmi,segi
    ret = {"make":{}}
    for l in inputs:
        if "!" in l: l = l[:l.index("!")]
        if "#" in l: l = l[:l.index("#")]
        s = l.split()
        ntok = len(s)
        if ntok == 0: continue
        if s[0].lower().startswith("make"):
            itk = 1
            r = ret["make"]
            while itk < ntok:
                if s[itk].lower().startswith("hydr"): #default a
                    r["hydr"] = s[itk+1][0].lower()
                    if r["hydr"] not in "yanf":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("chec"): # default n?
                    tmp = s[itk+1].lower()
                    if tmp.startswith(("none", "0")):
                        r["check"] = "0"
                    elif tmp.startswith(("liga", "n")):
                        r["check"] = "n"
                    elif tmp.startswith(("all", "y")):
                        r["check"] = "y"
                    else:
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("newl"): #default e
                    tmp = s[itk+1].lower()
                    if tmp.startswith("e"): # exit
                        r["newligand"] = False
                    elif tmp.startswith(("c", "y", "noex")): # noexit
                        r["newligand"] = True
                    else:
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("buil"): #default n
                    r["build"] = s[itk+1][0].lower()
                    if r["build"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("pept"): # default n
                    r["pept"] = s[itk+1][0].lower()
                    if r["pept"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("link"):
                    r["link"] = s[itk+1][0].lower()
                    if r["link"] not in "ynd0": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("suga"):
                    r["sugar"] = s[itk+1][0].lower()
                    if r["sugar"] not in "ynds": # what is s?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("conn"): # TODO read conn_tolerance? (make conn tole val)
                    r["conn"] = s[itk+1][0].lower()
                    if r["conn"] not in "ynd0": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("symm"):
                    r["symm"] = s[itk+1][0].lower()
                    if r["symm"] not in "ync": # what is 0?
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("chai"):
                    r["chain"] = s[itk+1][0].lower()
                    if r["chain"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("cisp"):
                    r["cispept"] = s[itk+1][0].lower()
                    if r["cispept"] not in "yn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("ss"):
                    r["ss"] = s[itk+1][0].lower()
                    if r["ss"] not in "ydn":
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif s[itk].lower().startswith("exit"):
                    r["exit"] = s[itk+1][0].lower() == "y" # TODO refmac works just with "make exit"
                    itk += 2
                else:
                    itk += 1
        elif s[0].lower().startswith(("sour", "scat")):
            k = s[1].lower()
            if k.startswith("em"):
                ret["source"] = "em"
            elif k.startswith("e"):
                ret["source"] = "ec"
            elif k.startswith("n"):
                ret["source"] = "ne"
            else:
                ret["source"] = "xr"
            # TODO check mb, lamb
        elif s[0].lower().startswith("refi"): # TODO support this. Note that only valid with hklin
            itk = 1
            while itk < ntok:
                if s[itk].startswith("type"):
                    if itk+1 < ntok and s[itk+1].startswith("unre"):
                        ret["refi"] = {"type": "unre"}
                    itk += 2
                else:
                    itk += 1
    def sorry(s): raise SystemExit("Sorry, {} is not supported".format(s))
    if ret["make"].get("hydr") == "f":
        sorry("make hydr full")
    if ret["make"].get("buil") == "y":
        sorry("make build yes")
    return ret
# parse_keywords()

def prepare_crd(xyzin, crdout, ligand, make, monlib_path=None, h_pos="elec", auto_box_with_padding=None,
                no_adjust_hydrogen_distances=False):
    assert h_pos in ("elec", "nucl")
    h_change = dict(a=gemmi.HydrogenChange.ReAddButWater,
                    y=gemmi.HydrogenChange.NoChange,
                    n=gemmi.HydrogenChange.Remove)[make.get("hydr", "a")]
    st = utils.fileio.read_structure(xyzin)
    if not st.cell.is_crystal():
        if auto_box_with_padding is not None:
            st.cell = utils.model.box_from_model(st[0], auto_box_with_padding)
            st.spacegroup_hm = "P 1"
            logger.writeln("Box size from the model with padding of {}: {}".format(auto_box_with_padding, st.cell.parameters))
        else:
            raise SystemExit("Error: unit cell is not defined in the model.")


    # we do not have DOD. will not change ND4->NH4 and SPW->SPK, as hydrogen atom names are different
    for chain in st[0]:
        for res in chain:
            if res.name == "DOD":
                logger.writeln("Warning: changing DOD to HOH (chain {} residue {})".format(chain.name, res.seqid))
                res.name = "HOH"

    # TODO read dictionary from xyzin (priority: user cif -> monlib -> xyzin
    st.entities.clear()
    st.setup_entities()

    try:
        monlib = utils.restraints.load_monomer_library(st,
                                                       monomer_dir=monlib_path,
                                                       cif_files=ligand,
                                                       stop_for_unknowns=not make.get("newligand"))
    except RuntimeError as e:
        raise SystemExit("Error: {}".format(e))

    if make.get("cispept", "y") == "y": st.assign_cis_flags()
    if make.get("link", "n") == "y": # TODO support it correctly, and also make link define. what is 0?
        #utils.restraints.find_and_fix_links(st, monlib) # TODO fix for unknown links
        logger.writeln("Make link yes specified. Finding links..")
        before = len(st.connections)
        gemmi.add_automatic_links(st[0], st, monlib)
        for i in range(before, len(st.connections)):
            con = st.connections[i]
            logger.writeln(" automatic link: {} - {} id= {}".format(con.partner1, con.partner2, con.link_id))

    refmac_fixes = None
    max_seq_num = max([max(res.seqid.num for res in chain) for model in st for chain in model])
    if max_seq_num > 9999:
        logger.writeln("Max residue number ({}) exceeds 9999. Needs workaround.".format(max_seq_num))
        topo = gemmi.prepare_topology(st, monlib, ignore_unknown_links=True)
        refmac_fixes = utils.refmac.FixForRefmac(st, topo, 
                                                 fix_microheterogeneity=False,
                                                 fix_resimax=True,
                                                 fix_nonpolymer=False)

    if make.get("hydr") == "a": logger.writeln("(re)generating hydrogen atoms")
    topo = gemmi.prepare_topology(st, monlib, h_change=h_change, warnings=logger, reorder=True, ignore_unknown_links=False) # we should remove logger here??
    if make.get("hydr") != "n" and st[0].has_hydrogen():
        if h_pos == "nucl" and (make.get("hydr") == "a" or not no_adjust_hydrogen_distances):
            resnames = st[0].get_all_residue_names()
            utils.restraints.check_monlib_support_nucleus_distances(monlib, resnames)
            logger.writeln("adjusting hydrogen position to nucleus")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.Nucleus, default_scale=1.1)
        elif h_pos == "elec" and make.get("hydr") == "y" and not no_adjust_hydrogen_distances:
            logger.writeln("adjusting hydrogen position to electron cloud")
            topo.adjust_hydrogen_distances(gemmi.Restraints.DistanceOf.ElectronCloud)

    doc = gemmi.prepare_refmac_crd(st, topo, monlib, h_change)
    doc.write_file(crdout, style=gemmi.cif.Style.NoBlankLines)
    logger.writeln("crd file written: {}".format(crdout))
    return refmac_fixes
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

def modify_output(pdbout, cifout, fixes):
    utils.fileio.rotate_file(pdbout, copy=True)
    utils.fileio.rotate_file(cifout, copy=True)
    
    st = utils.fileio.read_structure(cifout)
    st.raw_remarks = gemmi.read_pdb(pdbout).raw_remarks
    if fixes is not None:
        fixes.modify_back(st)

    utils.fileio.write_mmcif(st, cifout, cifout)
    
    chain_id_len_max = max([len(x) for x in utils.model.all_chain_ids(st)])
    seqnums = [res.seqid.num for chain in st[0] for res in chain]
    if chain_id_len_max > 1 or min(seqnums) <= -1000 or max(seqnums) >= 10000:
        logger.writeln("This structure cannot be saved as an official PDB format. Using hybrid-36. Header part may be inaccurate.")
    st.expand_hd_mixture()
    utils.fileio.write_pdb(st, pdbout)
# modify_output()

def main(args):
    args.ligand = sum(args.ligand, []) if args.ligand else []
    inputs = []
    for l in sys.stdin:
        inputs.append(l)
        if l.strip().lower() == "end":
            break

    xyzin, opts = get_opt(args.opts, "xyzin")
    xyzout = get_opt(args.opts, "xyzout", keep=True)[0]
    libin, _ = get_opt(opts, "libin") # print libin!!
    keywords = parse_keywords(inputs) # TODO expand @
    if libin: args.ligand.append(libin)

    # TODO what if restin is given or make cr prepared is given?
    # TODO check make pept/link/suga/ss/conn/symm/chain

    # Process model
    crdout = None
    refmac_fixes = None
    if xyzin is not None:
        #tmpfd, crdout = tempfile.mkstemp(prefix="gemmi_", suffix=".crd") # TODO use dir=CCP4_SCR
        #os.close(tmpfd)
        crdout = "gemmi_{}_{}.crd".format(utils.fileio.splitext(os.path.basename(xyzin))[0], os.getpid())
        refmac_fixes = prepare_crd(xyzin, crdout, args.ligand, make=keywords["make"], monlib_path=args.monlib,
                                   h_pos="nucl" if keywords.get("source")=="ne" else "elec",
                                   auto_box_with_padding=args.auto_box_with_padding,
                                   no_adjust_hydrogen_distances=args.no_adjust_hydrogen_distances)
        opts.extend(["xyzin", crdout])

    if keywords["make"].get("exit"):
        return
        
    # Run Refmac
    cmd = [args.exe] + opts
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
    for l in iter(p.stdout.readline, ""):
        logger.write(l)
    p.wait()

    # Modify output
    pdbout, cifout = get_output_model_names(xyzout)
    modify_output(pdbout, cifout, refmac_fixes)
# main()

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])
    main(args)
    
