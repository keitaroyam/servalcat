"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import subprocess
import pipes
import json
import copy
import re
import os
import string
import itertools
import tempfile
from servalcat.utils import logger
from servalcat.utils import fileio

re_version = re.compile("#.* Refmac *version ([^ ]+) ")
re_error = re.compile('(warn|error *[:]|error *==|^error)', re.IGNORECASE)
re_outlier_start = re.compile("\*\*\*\*.*outliers")

def check_version(exe="refmac5"):
    p = subprocess.Popen([exe], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    p.stdin.write("end\n")
    p.stdin.close()
    ver = ()
    for l in iter(p.stdout.readline, ""):
        r_ver = re_version.search(l)
        if r_ver:
            ver = tuple(map(int, r_ver.group(1).split(".")))
    p.wait()
    return ver
# check_version()

def ensure_ccp4scr():
    tmpdir = os.environ.get("CCP4_SCR")
    if tmpdir:
        if os.path.isdir(tmpdir): # TODO check writability
            try:
                t = tempfile.TemporaryFile(dir=tmpdir)
                t.close()
                return
            except OSError:
                logger.writeln("Warning: cannot write files in CCP4_SCR= {}".format(tmpdir))
        else:
            try:
                os.makedirs(tmpdir)
                return
            except:
                logger.writeln("Warning: cannot create CCP4_SCR= {}".format(tmpdir))

    os.environ["CCP4_SCR"] = tempfile.mkdtemp(prefix="ccp4tmp")
    logger.writeln("Updated CCP4_SCR= {}".format(os.environ["CCP4_SCR"]))
# ensure_ccp4scr()

def external_restraints_json_to_keywords(json_in):
    ret = []
    with open(json_in) as f: exte_list = json.load(f)
    for e in exte_list:
        if "use" in e:
            ret.append("EXTERNAL USE {}".format(e["use"]))
        if "dmax" in e:
            ret.append("EXTERNAL DMAX {0}".format(e["dmax"]))
        if "weight_scale" in e:
            ret.append("EXTERNAL WEIGHT SCALE {0}".format(e["weight_scale"]))
        if "weight_gmwt" in e:
            ret.append("EXTERNAL WEIGHT GMWT {0}".format(e["weight_gmwt"]))
        if "file" in e:
            ret.append("@"+e["file"])

    return "\n".join(ret) + "\n"
# external_restraints_json_to_keywords()

def parse_atom_spec(s, itk):
    # s: list of keywords
    ret = {}
    while itk < len(s):
        if s[itk].lower().startswith(("chai", "segm")):
            ret["chain"] = s[itk+1]
            itk += 2
        elif s[itk].lower().startswith("resi"):
            ret["resi"] = int(s[itk+1])
            itk += 2
        elif s[itk].lower().startswith("ins"):
            ret["icode"] = s[itk+1] if s[itk+1] != "." else " "
            itk += 2
        elif s[itk].lower().startswith(("atom", "atna", "name")):
            if s[itk+1] == "{":
                idx_close = s[itk+1:].index("}") + itk + 1
                ret["names"] = s[itk+2:idx_close]
                itk = idx_close + 1
            else:
                ret["names"] = [s[itk+1]]
                itk += 2
        elif s[itk].lower().startswith("alt"):
            ret["altloc"] = s[itk+1]
            itk += 2
        elif s[itk].lower().startswith("symm"):
            ret["symm"] = s[itk+1][0].lower() == "y"
        else:
            break

    return ret, itk
# parse_atom_spec()

def read_exte_line(l):
    # using the same variable names as used in read_extra_restraints.f
    s = l.split()
    ret = {}
    rest_flag_old = rest_flag = False
    if s[0].lower().startswith("exte"):
        if s[1].lower().startswith("gene"): return ret
        elif s[1].lower().startswith("file"): # XXX not supported
            file_ext_now = s[2]
        elif s[1].lower().startswith("syma"): # symall
            # refmac sets "n" by default if "syma" given - but it is not good!
            ret["symall_block"] = "y" if s[2][0].lower() == "y" else "n"
            if s[3].lower().startswith("excl"): # exclude
                ret["exclude_self_block"] = "s" if s[4].lower().startswith("self") else "n"
        elif s[1].lower().startswith("typa"): # typeall
            #type_default = 2
            ret["type_default"] = max(0, min(2, int(s[2])))
        elif s[1].lower().startswith("alph"): # alphall
            ret["alpha_default"] = float(s[2])
        elif s[1].lower().startswith("verb"): # verbose
            ret["ext_verbose"] = s[2][0].lower() != "n" and not s[2].lower().startswith("off")
            # print("External verbose is on, i.e.") ...
        elif s[1].lower().startswith("weig"): # weight
            ret["weight"] = {}
            itk = 2
            while itk < len(s): # FIXME check out-of-bounds in s[]
                if s[itk].lower().startswith("scal"):
                    ret["weight"]["scale"] = {}
                    try:
                        ret["weight"]["scale"]["loc"] = float(s[itk+1]) # scale_sigma_loc
                        itk += 2
                    except ValueError:
                        pass
                    for k in ("angl", "tors", "chir", "plan", "dist", "inte"):
                        if s[itk+1].lower().startswith(k):
                            ret["weight"]["scale"][k]  = float(s[itk+2])
                            itk += 3
                elif s[itk].lower().startswith("sgmn"):
                    ret["weight"]["sigma_min_loc"] = float(s[itk+1])
                    itk += 2
                elif s[itk].lower().startswith("sgmx"):
                    ret["weight"]["sigma_max_loc"] = float(s[itk+1])
                    itk += 2
                else:
                    raise RuntimeError("Error==> EXTE keyword interpretation: {}".format(l))
        elif s[1].lower().startswith(("miss", "unde")): # undefined
            ret["ignore_undefined"] = s[2].lower().startswith("igno")
        elif s[1].lower().startswith("hydr"): # hydrogen
            ret["ignore_hydrogens"] = s[2].lower().startswith("igno")
        elif s[1].lower().startswith("cut"):
            ret["sd_ext_cut"] = float(s[2])
        elif s[1].lower().startswith("dmax"):
            ret["dist_max_external"] = float(s[2])
        elif s[1].lower().startswith("dmin"):
            ret["dist_min_external"] = float(s[2])
        elif s[1].lower().startswith("use"):
            ret["use_atoms"] = s[2][0].lower()
            if ret["use_atoms"] not in ("a", "m", "h"):
                logger.writeln("invalid exte use keyword: {}".format(s[2]))
                ret["use_atoms"] = "a"
        elif s[1].lower().startswith("conv"):
            if s[2].lower().startswith("pref"):
                ret["prefix_ch"] = s[3]
        elif s[1].lower().startswith(("dist", "plan", "chir", "angl", "inte", "tors")):
            ret["rest_type"] = s[1][:4].lower()
            itk = 2
            iat = 0
            ret["itype_in"] = 1
            ret["restr"] = {}
            ret["restr"]["specs"] = [None for _ in range(4)] # up to 4
            while itk < len(s):
                if s[itk].lower().startswith(("firs", "seco", "thir", "four", "next", "atre", "atin")):
                    iat = dict(firs=0, seco=1, thir=2, four=3).get(s[itk][:4].lower(), iat+1)
                    atoms, itk = parse_atom_spec(s, itk+1)
                    ret["restr"]["specs"][iat] = atoms
                elif s[itk].lower().startswith("type"):
                    try:
                        ret["itype_in"] = int(s[itk+1])
                    except ValueError:
                        ret["itype_in"] = dict(o=0, f=2).get(s[itk+1][0].lower(), 1)
                    if not (0 <= ret["itype_in"] <= 2):
                        logger.writeln("WARNING: wrong type is given. setting to 2.\n=> {}".format(l))
                        ret["itype_in"] = 2
                    itk += 2
                elif s[itk].lower().startswith("symm"):
                    ret["symm_in"] = s[itk+1][0].lower() == "y"
                    itk += 2
                else:
                    d = dict(valu="dist_value", dmin="dmin", dmax="dmax", smin="smin_value", smax="smax_value",
                             sigm="sigma_value", alph="alpha_in", prob="prob_in")
                    k = s[itk][:4].lower()
                    if k in d:
                        ret["restr"][d[k]] = float(s[itk+1])
                        itk += 2
                    else:
                        logger.writeln("unrecognised key: {}\n=> {}".format(s[itk], l))
        elif s[1].lower().startswith("stac"):
            ret["rest_type"] = "stac"
            ret["restr"] = {}
            ret["restr"]["specs"] = [[] for _ in range(2)]
            itk = 2
            #if s[itk].lower().startswith("dist"):
            ip = 0
            while itk < len(s):
                if s[itk].lower().startswith("plan"):
                    ip = int(s[itk+1])
                    itk += 2
                    if ip not in (1, 2):
                        raise RuntimeError("Problem with stacking instructions. Plane number can be 1 or 2.\n=> {}".format(l))
                elif s[itk].lower().startswith(("firs", "next")):
                    atoms, itk = parse_atom_spec(s, itk+1)
                    ret["restr"]["specs"][ip-1] = atoms
                elif s[itk].lower().startswith(("dist", "sddi", "angl", "sdan", "type")):
                    k = dict(dist="dist_id", sddi="dist_sd", angl="angle_id", sdan="angle_sd", type="type_r")[s[itk][:4].lower()]
                    ret["restr"][k] = float(s[itk+1]) if k != "type_r" else int(s[itk+1])
                    itk += 2
                else:
                    logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[itk], l))
                    itk += 1
        else:
            logger.writeln("WARNING: cannot parse: {}".format(l))
    return ret
# read_exte_line

def read_tls_file(tlsin):
    # TODO sort out L/S units - currently use Refmac tlsin/out as is
    # TODO change to gemmi::TlsGroup?
    
    groups = []
    with open(tlsin) as ifs:
        for l in ifs:
            l = l.strip()
            if l.startswith("TLS"):
                title = l[4:]
                groups.append(dict(title=title, ranges=[], origin=None, T=None, L=None, S=None))
            elif l.startswith("RANG"):
                r = l[l.index(" "):].strip()
                groups[-1]["ranges"].append(r)
            elif l.startswith("ORIG"):
                try:
                    groups[-1]["origin"] = gemmi.Position(*(float(x) for x in l.split()[1:]))
                except:
                    raise ValueError("Prolem with TLS file: {}".format(l))
            elif l.startswith("T   "):
                try:
                    groups[-1]["T"] = [float(x) for x in l.split()[1:7]]
                except:
                    raise ValueError("Prolem with TLS file: {}".format(l))
            elif l.startswith("L   "):
                try:
                    groups[-1]["L"] = [float(x) for x in l.split()[1:7]]
                except:
                    raise ValueError("Prolem with TLS file: {}".format(l))
            elif l.startswith("S   "):
                try:
                    groups[-1]["S"] = [float(x) for x in l.split()[1:10]]
                except:
                    raise ValueError("Prolem with TLS file: {}".format(l))

    return groups
# read_tls_file()

def write_tls_file(groups, tlsout):
    with open(tlsout, "w") as f:
        for g in groups:
            f.write("TLS {}\n".format(g["title"]))
            for r in g["ranges"]:
                f.write("RANGE {}\n".format(r))
            if g["origin"] is not None:
                f.write("ORIGIN ")
                f.write(" ".join("{:8.4f}".format(x) for x in g["origin"].tolist()))
                f.write("\n")
            for k in "TLS":
                if g[k] is not None:
                    f.write("{:4s}".format(k))
                    f.write(" ".join("{:8.4f}".format(x) for x in g[k]))
                    f.write("\n")
# write_tls_file()

class FixForRefmac:
    """
    Workaround for Refmac limitations
    - microheterogeneity
    - residue number > 9999

    XXX fix external restraints accordingly
    TODO fix _struct_conf, _struct_sheet_range, _pdbx_struct_sheet_hbond
    """
    def __init__(self, st, topo, fix_microheterogeneity=True, fix_resimax=True, fix_nonpolymer=True, add_gaps=False):
        self.MAXNUM = 9999
        self.fixes = []
        self.chainids = set(chain.name for chain in st[0])
        if fix_microheterogeneity:
            self.fix_microheterogeneity(st, topo)
        if add_gaps:
            self.add_gaps(st, topo)
        if fix_resimax: # This modifies chains, so topo will be broken
            self.fix_too_large_seqnum(st, topo)
        if fix_nonpolymer: # This modifies chains, so topo will be broken
            self.fix_nonpolymer(st)

    def new_chain_id(self, original_chain_id):
        # decide new chain ID
        for i in itertools.count(start=1):
            new_id = "{}{}".format(original_chain_id, i)
            if new_id not in self.chainids:
                self.chainids.add(new_id)
                return new_id

    def fix_metadata(self, st, changedict):
        # fix connections
        # changedict = dict(changes)
        aa2tuple = lambda aa: (aa.chain_name, aa.res_id.seqid.num, chr(ord(aa.res_id.seqid.icode)|0x20))
        for con in st.connections:
            for aa in (con.partner1, con.partner2):
                changeto = changedict.get(aa2tuple(aa))
                if changeto is not None:
                    aa.chain_name = changeto[0]
                    aa.res_id.seqid.num = changeto[1]
                    aa.res_id.seqid.icode = changeto[2]

    def add_gaps(self, st, topo):
        # Refmac (as of 5.8.0352) has a bug that makes two links for IAS (IAS-pept and usual TRANS/CIS)
        # However this implementation is even more harmful.. if gap is inserted to real gaps then necessary p link is also gone!
        for chain in st[0]:
            rs = chain.get_polymer()
            for i in range(1, len(rs)):
                res0 = rs[i-1]
                res = rs[i]
                links = topo.links_to_previous(res)
                if len(links) == 0 or links[0].link_id in ("gap", "?"):
                    con = gemmi.Connection()
                    con.asu = gemmi.Asu.Same
                    con.type = gemmi.ConnectionType.Unknown
                    con.link_id = "gap"
                    con.partner1 = gemmi.AtomAddress(chain.name, res0.seqid, res0.name, "", "\0")
                    con.partner2 = gemmi.AtomAddress(chain.name, res.seqid, res.name, "", "\0")
                    logger.writeln("Refmac workaround (gap link): {}".format(con))
                    st.connections.append(con)

    def fix_microheterogeneity(self, st, topo):
        mh_res = []
        chains = []
        icodes = {} # to avoid overlaps
        modifications = [] # return value

        # Check if microheterogeneity exists
        for chain in st[0]:
            for rg in chain.get_polymer().residue_groups():
                if len(rg) > 1:
                    ress = [r for r in rg]
                    chains.append(chain.name)
                    mh_res.append(ress)
                    ress_str = "/".join([str(r) for r in ress])
                    logger.writeln("Microheterogeneity detected in chain {}: {}".format(chain.name, ress_str))

        if not mh_res: return

        for chain in st[0]:
            for res in chain:
                if res.seqid.icode != " ":
                    icodes.setdefault(chain.name, {}).setdefault(res.seqid.num, []).append(res.seqid.icode)

        def append_links(bond, prr, toappend):
            atoms = bond.atoms
            assert len(atoms) == 2
            found = None
            for i in range(2):
                if any(filter(lambda ra: atoms[i]==ra, prr)): found = i
            if found is not None:
                toappend.append([atoms[i], atoms[1-i]]) # prev atom, current atom
        # append_links()

        mh_res_all = sum(mh_res, [])
        mh_link = {}

        # Check links
        for chain in st[0]:
            for res in chain:
                # If this residue is microheterogeneous
                if res in mh_res_all:
                    for link in topo.links_to_previous(res):
                        mh_link.setdefault(id(res), []).append([link.res1, "prev", link.link_id, []])
                        append_links(topo.first_bond_in_link(link), link.res1, mh_link[id(res)][-1][-1])

                # Check if previous residue(s) is microheterogeneous
                for link in topo.links_to_previous(res):
                    prr = link.res1
                    if prr in mh_res_all:
                        mh_link.setdefault(id(prr), []).append([res, "next", link.link_id, []])
                        append_links(topo.first_bond_in_link(link), prr, mh_link[id(prr)][-1][-1])

        # Change IDs
        for chain_name, rr in zip(chains, mh_res):
            chars = string.ascii_uppercase
            # avoid already used inscodes
            if chain_name in icodes and rr[0].seqid.num in icodes[chain_name]:
                used_codes = set(icodes[chain_name][rr[0].seqid.num])
                chars = list(filter(lambda x: x not in used_codes, chars))
            for ir, r in enumerate(rr[1:]):
                modifications.append([(chain_name, r.seqid.num, r.seqid.icode),
                                      (chain_name, r.seqid.num, chars[ir])])
                r.seqid.icode = chars[ir]

        logger.writeln("DEBUG: mh_link= {}".format(mh_link))
        # Update connections (LINKR)
        for chain_name, rr in zip(chains, mh_res):
            for r in rr:
                for p in mh_link.get(id(r), []):
                    for atoms in p[-1]:
                        con = gemmi.Connection()
                        con.asu = gemmi.Asu.Same
                        con.type = gemmi.ConnectionType.Covale
                        con.link_id = p[2]
                        if p[1] == "prev":
                            p1 = gemmi.AtomAddress(chain_name, p[0].seqid, p[0].name, atoms[1].name, atoms[1].altloc)
                            p2 = gemmi.AtomAddress(chain_name, r.seqid, r.name, atoms[0].name, atoms[0].altloc)
                        else:
                            p1 = gemmi.AtomAddress(chain_name, r.seqid, r.name, atoms[1].name, atoms[1].altloc)
                            p2 = gemmi.AtomAddress(chain_name, p[0].seqid, p[0].name, atoms[0].name, atoms[0].altloc)

                        con.partner1 = p1
                        con.partner2 = p2
                        logger.writeln(" Adding link: {}".format(con))
                        st.connections.append(con)
            for r1, r2 in itertools.combinations(rr, 2):
                for a1 in set([a.altloc for a in r1]):
                    for a2 in set([a.altloc for a in r2]):
                        con = gemmi.Connection()
                        con.asu = gemmi.Asu.Same
                        con.link_id = "gap"
                        # XXX altloc will be ignored when atom does not match.. grrr
                        con.partner1 = gemmi.AtomAddress(chain_name, r1.seqid, r1.name, "", a1)
                        con.partner2 = gemmi.AtomAddress(chain_name, r2.seqid, r2.name, "", a2)
                        st.connections.append(con)

        self.fixes.append(modifications)
    # fix_microheterogeneity()

    def fix_nonpolymer(self, st):
        # Refmac (as of 5.8.0352) has a bug that links non-neighbouring nucleotides
        # It only happens with mmCIF file
        newchains = []
        changes = []
        for chain in st[0]:
            polymer = chain.get_polymer()
            if len(polymer) == len(chain): continue
            if len(polymer) == 0: continue
            del_idxes = []
            newchains.append(gemmi.Chain(self.new_chain_id(chain.name)))
            logger.writeln("Refmac workaround (nonpolymer-fix) {} => {} ({} residues)".format(chain.name, newchains[-1].name,
                                                                                            len(chain) - len(polymer)))
            for i, res in enumerate(chain):
                if res in polymer: continue
                newchains[-1].add_residue(res)
                del_idxes.append(i)
                changes.append([(chain.name, res.seqid.num, res.seqid.icode),
                                (newchains[-1].name, newchains[-1][-1].seqid.num, newchains[-1][-1].seqid.icode)])
            for i in reversed(del_idxes):
                del chain[i]
                
        for c in newchains:
            st[0].add_chain(c)
        if changes:
            st.remove_empty_chains()
            self.fix_metadata(st, dict(changes))
        self.fixes.append(changes)

    def fix_too_large_seqnum(self, st, topo):
        # Refmac cannot handle residue id > 9999
        # What to do:
        # - move to new chains
        # - modify link records (and others?)
        # - add link record if needed
        newchains = []
        changes = []
        
        for chain in st[0]:
            maxseqnum = max([r.seqid.num for r in chain])
            if maxseqnum > self.MAXNUM:
                offset = 0
                #target = [res for res in chain if res.seqid.num > 9999]
                del_idxes = []
                for ires, res in enumerate(chain):
                    if res.seqid.num <= self.MAXNUM: continue
                    if res.seqid.num - offset > self.MAXNUM:
                        newchains.append(gemmi.Chain(self.new_chain_id(chain.name)))
                        offset = res.seqid.num - 1
                        # need to keep link to previous residue if exists
                        for link in topo.links_to_previous(res):
                            logger.writeln("Link: {} {} {} alt= {} {}".format(link.link_id, link.res1, link.res2, 
                                                                            link.alt1, link.alt2))

                            con = gemmi.Connection()
                            con.type = gemmi.ConnectionType.Covale
                            con.link_id = link.link_id
                            #return link
                            bond = topo.first_bond_in_link(link)
                            if bond is not None:
                                con.partner1 = gemmi.AtomAddress(chain.name, link.res1.seqid, link.res1.name, bond.atoms[0].name, bond.atoms[0].altloc)
                                con.partner2 = gemmi.AtomAddress(chain.name, link.res2.seqid, link.res2.name, bond.atoms[1].name, bond.atoms[1].altloc)
                                st.connections.append(con)

                    newchains[-1].add_residue(res)
                    newchains[-1][-1].seqid.num -= offset
                    del_idxes.append(ires)
                    prev = chain[ires-1].seqid if ires > 0 else None
                    changes.append([(chain.name, res.seqid.num, res.seqid.icode),
                                    (newchains[-1].name, newchains[-1][-1].seqid.num, newchains[-1][-1].seqid.icode)])
                    logger.writeln("Refmac workaround (too large seq) {} => {} {}".format(changes[-1][0], changes[-1][1], res.name))

                for i in reversed(del_idxes):
                    del chain[i]

        for c in newchains:
            st[0].add_chain(c)
        if changes:
            st.remove_empty_chains()
            self.fix_metadata(st, dict(changes))
        self.fixes.append(changes)

    def fix_model(self, st, changedict):
        chain_newid = set()
        for chain in st[0]:
            for res in chain:
                changeto = changedict.get((chain.name, res.seqid.num, res.seqid.icode))
                if changeto is not None:
                    logger.writeln("back: {} {} to {}".format(chain.name, res.seqid, changeto))
                    #chain.name = changeto[0] # this is ok when modify back
                    chain_newid.add((chain, changeto[0]))
                    res.seqid.num = changeto[1]
                    res.seqid.icode = changeto[2]

        for chain, newid in chain_newid:
            chain.name = newid
        st.merge_chain_parts()
        self.fix_metadata(st, changedict)
        
    def modify_back(self, st):
        for fix in reversed(self.fixes):
            reschanges = dict([x[::-1] for x in fix])
            self.fix_model(st, reschanges)
        
        
class Refmac:
    def __init__(self, **kwargs):
        self.prefix = "refmac"
        self.hklin = self.xyzin = ""
        self.source = "electron"
        self.lab_f = None
        self.lab_sigf = None
        self.lab_phi = None
        self.libin = None
        self.tlsin = None
        self.hydrogen = "all"
        self.hout = False
        self.ncycle = 10
        self.tlscycle = 0
        self.resolution = None
        self.weight_matrix = None
        self.weight_auto_scale = None
        self.bfactor = None
        self.jellybody = None
        self.jellybody_sigma, self.jellybody_dmax = 0.01, 4.2
        self.ncsr = None
        self.shake = None
        self.keyword_files = []
        self.keywords = []
        self.external_restraints_json = None
        self.exe = "refmac5"
        self.monlib_path = None
        self.keep_chain_ids = False
        self.show_log = False # summary only if false
        self.global_mode = kwargs.get("global_mode")
        
        for k in kwargs:
            if k == "args":
                self.init_from_args(kwargs["args"])
            else:
                setattr(self, k, kwargs[k])

        ensure_ccp4scr()
    # __init__()

    def init_from_args(self, args):
        self.hklin = args.mtz
        self.xyzin = args.model
        self.libin = args.ligand
        self.tlsin = args.tlsin
        self.ncycle = args.ncycle
        self.tlscycle = args.tlscycle
        self.lab_f = args.lab_f
        self.lab_phi = args.lab_phi
        self.lab_sigf = args.lab_sigf
        self.hydrogen = args.hydrogen
        self.hout = args.hout
        self.ncsr = args.ncsr
        self.bfactor = args.bfactor
        self.jellybody = args.jellybody
        self.jellybody_sigma, self.jellybody_dmax = args.jellybody_params
        self.resolution = args.resolution
        self.weight_auto_scale = args.weight_auto_scale
        self.keyword_files = args.keyword_file
        self.keywords = args.keywords
        self.external_restraints_json = args.external_restraints_json
        self.exe = args.exe
        self.show_log = args.show_refmac_log
        self.monlib_path = args.monlib
    # init_from_args()

    def copy(self, **kwargs):
        ret = copy.deepcopy(self)
        for k in kwargs:
            setattr(ret, k, kwargs[k])
            
        return ret
    # copy()

    def set_libin(self, ligands):
        if not ligands: return
        if len(ligands) > 1:
            mcif = "merged_ligands.cif" # XXX directory!
            logger.writeln("Merging ligand cif files: {}".format(ligands))
            fileio.merge_ligand_cif(ligands, mcif)
            self.libin = mcif
        else:
            self.libin = ligands[0]
    # set_libin()
            
    
    def make_keywords(self):
        ret = ""
        labin = []
        if self.lab_f: labin.append("FP={}".format(self.lab_f))
        if self.lab_sigf: labin.append("SIGFP={}".format(self.lab_sigf))
        if self.lab_phi: labin.append("PHIB={}".format(self.lab_phi))
        if labin:
            ret += "labin {}\n".format(" ".join(labin))


        ret += "make hydr {}\n".format(self.hydrogen)
        ret += "make hout {}\n".format("yes" if self.hout else "no")
        
        if self.global_mode == "spa":
            ret += "solvent no\n"
            ret += "scale lssc isot\n"
            ret += "source em mb\n"
        elif self.source == "electron":
            ret += "source ec mb\n"
        elif self.source == "neutron":
            ret += "source n\n"

        ret += "ncycle {}\n".format(self.ncycle)
        if self.resolution is not None:
            ret += "reso {}\n".format(self.resolution)
        if self.weight_matrix is not None:
            ret += "weight matrix {}\n".format(self.weight_matrix)
        elif self.weight_auto_scale is not None:
            ret += "weight auto {:.2e}\n".format(self.weight_auto_scale)
        else:
            ret += "weight auto\n"

        if self.bfactor is not None:
            ret += "bfactor set {}\n".format(self.bfactor)
        if self.jellybody:
            ret += "ridge dist sigma {:.3e}\n".format(self.jellybody_sigma)
            ret += "ridge dist dmax {:.2e}\n".format(self.jellybody_dmax)
        if self.ncsr:
            ret += "ncsr {}\n".format(self.ncsr)
        if self.shake:
            ret += "rand {}\n".format(self.shake)
        if self.tlscycle > 0:
            ret += "refi tlsc {}\n".format(self.tlscycle)
            ret += "tlsout addu\n"
        if self.keep_chain_ids:
            ret += "pdbo keep auth\n"

        if self.external_restraints_json:
            ret += external_restraints_json_to_keywords(self.external_restraints_json)
            
        if self.keyword_files:
            for f in self.keyword_files:
                ret += "@{}\n".format(f)

        if self.keywords:
            ret += "\n".join(self.keywords).strip() + "\n"

        return ret
    # make_keywords()

    def xyzout(self): return self.prefix + ".pdb"
    def hklout(self): return self.prefix + ".mtz"
    def tlsout(self): return self.prefix + ".tls"

    def make_cmd(self):
        cmd = [self.exe]
        cmd.extend(["hklin", self.hklin])
        cmd.extend(["hklout", self.hklout()])
        cmd.extend(["xyzin", self.xyzin])
        cmd.extend(["xyzout", self.xyzout()])
        if self.libin:
            cmd.extend(["libin", self.libin])
        if self.tlsin:
            cmd.extend(["tlsin", self.tlsin])
        if self.tlscycle > 0:
            cmd.extend(["tlsout", self.tlsout()])
        if self.source == "neutron":
            cmd.extend(["atomsf", os.path.join(os.environ["CLIBD"], "atomsf_neutron.lib")])
            
        return cmd
    # make_cmd()

    def run_refmac(self):
        cmd = self.make_cmd()
        stdin = self.make_keywords()
        with open(self.prefix+".inp", "w") as ofs: ofs.write(stdin)

        logger.writeln("Running REFMAC5..")
        logger.writeln("{} <<__eof__ > {}".format(" ".join(pipes.quote(x) for x in cmd), self.prefix+".log"))
        logger.write(stdin)
        logger.writeln("__eof__")

        env = os.environ
        if self.monlib_path: env["CLIBD_MON"] = os.path.join(self.monlib_path, "") # should end with /
        
        p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, env=env)
        p.stdin.write(stdin)
        p.stdin.close()

        log = open(self.prefix+".log", "w")
        cycle = 0
        re_cycle_table = re.compile("Cycle *([0-9]+). Rfactor analysis")
        re_actual_weight = re.compile("Actual weight *([^ ]+) *is applied to the X-ray term")
        rmsbond = ""
        rmsangle = ""
        log_delay = []
        summary_write = (lambda x: log_delay.append(x)) if self.show_log else logger.writeln
        outlier_flag = False
        last_table_flag = False
        last_table_keys = []
        occ_flag = False
        occ_cycles = 0
        ret = {"version":None,
               "cycles": [{"cycle":i} for i in range(self.ncycle+self.tlscycle+1)],
               } # metadata
        
        for l in iter(p.stdout.readline, ""):
            log.write(l)

            if self.show_log:
                print(l, end="")
            
            r_ver = re_version.search(l)
            if r_ver:
                ret["version"] = r_ver.group(1)
                summary_write("Starting Refmac {} (PID: {})".format(r_ver.group(1), p.pid))

            # print error/warning
            r_err = re_error.search(l)
            if r_err:
                if self.global_mode == "spa":
                    if "Figure of merit of phases has not been assigned" in l:
                        continue
                    elif "They will be assumed to be equal to 1.0" in l:
                        continue
                summary_write(l.rstrip())

            # print outliers
            r_outl = re_outlier_start.search(l)
            if r_outl:
                outlier_flag = True
                summary_write(l.rstrip())
            elif outlier_flag:
                if l.strip() == "" or "monitored" in l or "dev=" in l or "sigma=" in l.lower() or "sigma.=" in l:
                    summary_write(l.rstrip())
                else:
                    outlier_flag = False

            if "TLS refinement cycle" in l:
                cycle = int(l.split()[-1])
            elif "----Group occupancy refinement----" in l:
                occ_flag = True
                occ_cycles += 1
                cycle += 1
            elif "CGMAT cycle number =" in l:
                cycle = int(l[l.index("=")+1:]) + self.tlscycle + occ_cycles
                occ_flag = False
            
            r_cycle = re_cycle_table.search(l)
            if r_cycle: cycle = int(r_cycle.group(1))                

            for i in range(len(ret["cycles"]), cycle):
                ret["cycles"].append({"cycle":i})

            if "Overall R factor                     =" in l and cycle > 0:
                rfac = l[l.index("=")+1:].strip()
                if self.global_mode != "spa":
                    summary_write(" cycle= {:3d} R= {}".format(cycle-1, rfac))
            elif "Average Fourier shell correlation    =" in l and cycle > 0:
                fsc = l[l.index("=")+1:].strip()
                if occ_flag:
                    note = "(occupancy)"
                elif cycle == 1:
                    note = "(initial)"
                elif cycle <= self.tlscycle+1:
                    note = "(TLS)"
                elif cycle > self.ncycle + occ_cycles + self.tlscycle:
                    note = "(final)"
                else:
                    note = ""

                ret["cycles"][cycle-1]["fsc_average"] = fsc
                if self.global_mode == "spa":
                    summary_write(" cycle= {:3d} FSCaverage= {} {}".format(cycle-1, fsc, note))
            elif "Rms BondLength" in l:
                rmsbond = l
            elif "Rms BondAngle" in l:
                rmsangle = l

            r_actual_weight = re_actual_weight.search(l)
            if r_actual_weight:
                ret["cycles"][cycle-1]["actual_weight"] = r_actual_weight.group(1)

            # Final table
            if "    Ncyc    Rfact    Rfree     FOM" in l:
                last_table_flag = True
                last_table_keys = l.split()
                if last_table_keys[-1] == "$$": del last_table_keys[-1]
            elif last_table_flag:
                if "$$ Final results $$" in l:
                    last_table_flag = False
                    continue
                sp = l.split()
                if len(sp) == len(last_table_keys) and sp[0] != "$$":
                    cyc = int(sp[last_table_keys.index("Ncyc")])
                    key_name = dict(rmsBOND="rms_bond", zBOND="rmsz_bond",
                                    rmsANGL="rms_angle", zANGL="rmsz_angle",
                                    rmsCHIRAL="rms_chiral")
                    for k in key_name:
                        if k in last_table_keys:
                            ret["cycles"][cyc][key_name[k]] = sp[last_table_keys.index(k)]
                        else:
                            logger.error("table does not have key {}?".format(k))
                            
        retcode = p.wait()
        log.close()
        if log_delay:
            logger.writeln("== Summary of Refmac ==")
            logger.writeln("\n".join(log_delay))
                         
        if rmsbond:
            logger.writeln("                      Initial    Final")
            logger.writeln(rmsbond.rstrip())
            logger.writeln(rmsangle.rstrip())
            
        logger.writeln("REFMAC5 finished with exit code= {}".format(retcode))

        # TODO check timestamp
        if not os.path.isfile(self.xyzout()) or not os.path.isfile(self.hklout()):
            raise RuntimeError("REFMAC5 did not produce output files. Check {}".format(self.prefix+".log"))

        return ret
    # run_refmac()
# class Refmac
