"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat.utils import model as model_util
import gemmi
b_to_u = model_util.b_to_u

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
            itk += 2
        else:
            break

    return ret, itk
# parse_atom_spec()

def parse_from_to(s, itk):
    # s: list of keywords
    ret = {}
    if not s[itk].lower().startswith("from"):
        raise RuntimeError("invalid from_to instruction: {}".format(s))

    if s[itk+1] == "*":
        ret["ifirst"] = None # Refmac sets -9999
    else:
        ret["ifirst"] = int(s[itk+1])

    if s[itk+2].lower() != "to":
        ret["chain"] = s[itk+2]
        assert s[itk+3].lower() == "to"
        itk += 4
    else:
        itk += 3

    if s[itk] == "*":
        ret["ilast"] = None # Refmac sets 9999
    else:
        ret["ilast"] = int(s[itk])

    if "chain" not in ret:
        ret["chain"] = s[itk+1]
        itk += 2
    else:
        itk += 2
        
    return ret, itk
# parse_from_to()

def read_exte(s):
    # using the same variable names as used in read_extra_restraints.f
    ret = dict(defaults={})
    if not s: return ret
    defs = ret["defaults"]
    rest_flag_old = rest_flag = False
    if s[0].lower().startswith("exte"):
        if s[1].lower().startswith("gene"): return ret
        elif s[1].lower().startswith("file"): # XXX not supported
            file_ext_now = s[2]
        elif s[1].lower().startswith("syma"): # symall
            # refmac sets "n" by default if "syma" given - but it is not good!
            defs["symall_block"] = "y" if s[2][0].lower() == "y" else "n"
            if len(s) > 3 and s[3].lower().startswith("excl"): # exclude
                defs["exclude_self_block"] = s[4].lower().startswith("self")
        elif s[1].lower().startswith("typa"): # typeall
            #type_default = 2
            defs["type_default"] = max(0, min(2, int(s[2])))
        elif s[1].lower().startswith("alph"): # alphall
            defs["alpha_default"] = float(s[2])
        elif s[1].lower().startswith("verb"): # verbose
            defs["ext_verbose"] = s[2][0].lower() != "n" and not s[2].lower().startswith("off")
            # print("External verbose is on, i.e.") ...
        elif s[1].lower().startswith("weig"): # weight
            itk = 2
            while itk < len(s): # FIXME check out-of-bounds in s[]
                if s[itk].lower().startswith("scal"):
                    itk += 1
                    if itk >= len(s): break
                    try:
                        defs["scale_sigma_dist"] = float(s[itk]) # scale_sigma_loc
                        itk += 1
                        continue
                    except ValueError:
                        pass
                    for k in ("angl", "tors", "chir", "plan", "dist", "inte"):
                        if s[itk].lower().startswith(k):
                            itk += 1
                            if itk >= len(s): break
                            try:
                                defs["scale_sigma_{}".format(k)] = float(s[itk])
                                itk += 1
                                break
                            except ValueError:
                                pass
                elif s[itk].lower().startswith("sgmn"):
                    defs["sigma_min_loc"] = float(s[itk+1])
                    itk += 2
                elif s[itk].lower().startswith("sgmx"):
                    defs["sigma_max_loc"] = float(s[itk+1])
                    itk += 2
                else:
                    raise RuntimeError("Error==> EXTE keyword interpretation: {}".format(" ".join(s)))
        elif s[1].lower().startswith(("miss", "unde")): # undefined
            defs["ignore_undefined"] = s[2].lower().startswith("igno")
        elif s[1].lower().startswith("hydr"): # hydrogen
            defs["ignore_hydrogens"] = s[2].lower().startswith("igno")
        elif s[1].lower().startswith("cut"): # TODO I think this should be parsed outside
            ret["sd_ext_cut"] = float(s[2])  # as this affects everything (not only following blocks)
        elif s[1].lower().startswith("dmax"):
            defs["dist_max_external"] = float(s[2])
        elif s[1].lower().startswith("dmin"):
            defs["dist_min_external"] = float(s[2])
        elif s[1].lower().startswith("use"):
            defs["use_atoms"] = s[2][0].lower()
            if defs["use_atoms"] not in ("a", "m", "h"):
                logger.writeln("invalid exte use keyword: {}".format(s[2]))
                defs["use_atoms"] = "a"
        elif s[1].lower().startswith("conv"):
            if s[2].lower().startswith("pref"):
                defs["prefix_ch"] = s[3] # ???
        elif s[1].lower().startswith(("dist", "plan", "chir", "angl", "inte", "tors")):
            ret["rest_type"] = s[1][:4].lower()
            itk = 2
            iat = 0
            ret["restr"] = {}
            n_expect = dict(plan=0, dist=2, inte=2, angl=3).get(ret["rest_type"], 4)
            ret["restr"]["specs"] = [None for _ in range(n_expect)]
            while itk < len(s):
                if s[itk].lower().startswith(("firs", "seco", "thir", "four", "next", "atre", "atin")):
                    iat = dict(firs=0, seco=1, thir=2, four=3).get(s[itk][:4].lower(), iat+1)
                    atoms, itk = parse_atom_spec(s, itk+1)
                    if ret["rest_type"] == "plan":
                        ret["restr"]["specs"].append(atoms)
                    else:
                        ret["restr"]["specs"][iat] = atoms
                elif s[itk].lower().startswith("type"):
                    try:
                        ret["restr"]["itype_in"] = int(s[itk+1])
                    except ValueError:
                        ret["restr"]["itype_in"] = dict(o=0, f=2).get(s[itk+1][0].lower(), 1)
                    if not (0 <= ret["restr"]["itype_in"] <= 2):
                        logger.writeln("WARNING: wrong type is given. setting to 2.\n=> {}".format(" ".join(s)))
                        ret["restr"]["itype_in"] = 2
                    itk += 2
                elif s[itk].lower().startswith("symm"): # only for distance and angle
                    ret["restr"]["symm_in"] = s[itk+1][0].lower() == "y"
                    itk += 2
                else:
                    d = dict(valu="value", dmin="dmin", dmax="dmax", smin="smin_value", smax="smax_value",
                             sigm="sigma_value", alph="alpha_in", prob="prob_in")
                    k = s[itk][:4].lower()
                    if k in d:
                        ret["restr"][d[k]] = float(s[itk+1])
                        itk += 2
                    else:
                        logger.writeln("unrecognised key: {}\n=> {}".format(s[itk], " ".join(s)))
                        break
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
                        raise RuntimeError("Problem with stacking instructions. Plane number can be 1 or 2.\n=> {}".format(" ".join(s)))
                elif s[itk].lower().startswith(("firs", "next")):
                    atoms, itk = parse_atom_spec(s, itk+1)
                    ret["restr"]["specs"][ip-1] = atoms
                elif s[itk].lower().startswith(("dist", "sddi", "angl", "sdan", "type")):
                    k = dict(dist="dist_id", sddi="dist_sd", angl="angle_id", sdan="angle_sd", type="type_r")[s[itk][:4].lower()]
                    ret["restr"][k] = float(s[itk+1]) if k != "type_r" else int(s[itk+1])
                    itk += 2
                else:
                    logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[itk], " ".join(s)))
                    itk += 1
        elif s[1].lower().startswith(("harm", "spec")):
            ret["rest_type"] = s[1][:4].lower() # in Refmac, irest_type = 1 if harm else 2
            ret["restr"] = dict(rectype="", toler=0.5, sigma_t=0.5, sigma_u=2.0 * b_to_u, u_val_incl=False)
            itk = 2
            while itk < len(s):
                if s[itk].lower().startswith("auto"):
                    ret["restr"]["rectype"] = "auto"
                    itk += 1
                elif s[itk].lower().startswith("atin"):
                    ret["restr"]["rectype"] = "atom"
                    atoms, itk = parse_atom_spec(s, itk+1)
                    ret["restr"]["specs"] = [atoms]
                elif s[itk].lower().startswith("resi"):
                    ret["restr"]["rectype"] = "resi"
                    fromto, itk = parse_from_to(s, itk+1)
                    ret["restr"]["specs"] = [fromto]
                    if s[itk].lower().startswith("atom"):
                        ret["restr"]["specs"][0]["atom"] = s[itk+1] # called atom_resi in Refmac
                        itk += 2
                elif s[itk].lower().startswith("sigm"):
                    ret["restr"]["sigma_t"] = float(s[itk+1])
                    itk += 2
                elif s[itk].lower().startswith("tole"):
                    ret["restr"]["toler"] = float(s[itk+1])
                    itk += 2
                elif s[itk].lower().startswith(("uval", "bval")):
                    if len(s) > itk+1 and s[itk+1].lower().startswith("incl"):
                        ret["restr"]["u_val_incl"] = True
                        itk += 2
                    else:
                        ret["restr"]["u_val_incl"] = False
                        itk += 1
                elif s[itk].lower().startswith(("sigb", "sigu")):
                    ret["restr"]["sigma_u"] = float(s[itk+1]) * b_to_u
                    itk += 2
                else:
                    logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[itk], " ".join(s)))
                    itk += 1

        else:
            logger.writeln("WARNING: cannot parse: {}".format(" ".join(s)))
    return ret
# read_exte()

def read_ridge_params(l, r):
    s = l.split()
    assert s[0].lower().startswith("ridg")
    ntok = len(s)
    if s[1].lower().startswith("dist") and ntok > 2:
        if s[2].lower().startswith("with"):
            r.setdefault("groups", []).append({})
            #r["groups"][-1]["sigma"] = sigma_dist_r
            #r["groups"][-1]["dmax"] = dmax_dist_r
            itk = 3
            while itk < ntok:
                if s[itk].lower().startswith("chai"):
                    r["groups"][-1]["chain"] = s[itk+1]
                    itk += 2
                elif s[itk].lower().startswith("resi"):
                    r["groups"][-1]["resi"] = (int(s[itk+1]), int(s[itk+2]))
                    itk += 3
                elif s[itk].lower().startswith("sigm"):
                    v = float(s[itk+1])
                    if v < 0: v = 0.01
                    r["groups"][-1]["sigma"] = v
                    itk += 2
                elif s[itk].lower().startswith("dmax"):
                    v = float(s[itk+1])
                    if v < 0: v = 4.2
                    r["groups"][-1]["dmax"] = v
                    itk += 2
        elif s[2].lower().startswith("incl") and ntok > 3:
            # a: ridge_dist_include_all
            # h: ridge_dist_include_hbond
            # m: ridge_dist_include_main
            v = s[3][0].lower()
            if v in ("a", "h", "m"): r["include"] = v
        elif s[2].lower().startswith("sigm"):
            v = float(s[3])
            r["sigma"] = v if v > 0 else 0.01
        elif s[2].lower().startswith("dmax"):
            v = float(s[3])
            r["dmax"] = v if v > 0 else 4.2
        elif s[2].lower().startswith("inte") and ntok > 3:
            r["interchain"] = s[3][0].lower() == "y"
        elif s[2].lower().startswith("symm") and ntok > 3:
            r["intersym"] = s[3][0].lower() == "y"
        elif s[2].lower().startswith("long") and ntok > 3:
            r["long_range"] = max(0, int(s[3])) # long_range_residue_gap
        elif s[2].lower().startswith("shor") and ntok > 3:
            r["short_range"] = max(0, int(s[3])) # short_range_residue_gap
        elif s[2].lower().startswith("hydr"):
            r["hydrogen"] = ntok < 4 or s[3][0].lower() == "i" # hydrogens_include
        elif s[2].lower().startswith("side") and ntok > 3:
            r["sidechain"] = s[3][0].lower() == "i" # rb_side_chain_include
        elif s[2].lower().startswith("filt"):
            r["bvalue_filter"] = True
            if s[3].lower().startswith("bran"):
                v = float(s[4])
                r["bvalue_filter_range"] = v if v > 0 else 2.0
        else:
            logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[2], l))
    elif s[1].lower().startswith(("atom", "posi")): # not used
        r["sigma_pos"] = float(s[2]) if ntok > 2 else 0.1
    elif s[1].lower().startswith(("bval", "uval")) and ntok > 2:
        if s[2].lower().startswith("diff"):
            itk = 3
            while itk < ntok:
                if s[itk].lower().startswith("sigm"):
                    if ntok > itk + 1:
                        r["sigma_uval_diff"] = float(s[itk+1])
                        itk += 2
                    else:
                        r["sigma_uval_diff"] = 0.025
                        itk += 1
                elif s[itk].lower().startswith("dmax"): 
                    if ntok > itk + 1:
                        r["dmax_uval_diff"] = float(s[itk+1])
                        itk += 2
                    else:
                        r["dmax_uval_diff"] = 4.2
                        itk += 1
                elif s[itk].lower().startswith("dmwe"): # not used
                    if ntok > itk + 1:
                        r["dmax_uval_weight"] = float(s[itk+1]) * b_to_u
                        itk += 2
                    else:
                        r["dmax_uval_weight"] = 3.0
                        itk += 1
                else:
                    itk += 1
        else:
            r["sigma_b"] = float(s[2])
            r["sigma_u"] = float(s[2])
    else:
        logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[1], l))

    return r
# read_ridge_params()

def read_occupancy_params(l, r):
    s = l.split()
    if not s[0].lower().startswith("occu"):
        return r
    ntok = len(s)
    r.setdefault("groups", {}) # {igr: [{selection}, {selection}, ..]}
    r.setdefault("const", []) # [[is_comp, group_ids]]
    r.setdefault("ncycle", 0) # 0 means no refine
    if (ntok > 4 and
        s[1].lower().startswith("grou") and
        s[2].lower().startswith("id")):
        igr = s[3]
        gr = r["groups"].setdefault(igr, [])
        gr.append({})
        itk = 4
        while itk < ntok:
            if s[itk].lower().startswith(("chai", "segm")):
                gr[-1]["chains"] = []
                itk += 1
                while itk < ntok and not s[itk].lower().startswith(("resi","atom","alt")):
                    gr[-1]["chains"].append(s[itk])
                    itk += 1
            elif s[itk].lower().startswith("resi"):
                if s[itk+1].lower().startswith("from"):
                    gr[-1]["resi_from"] = gemmi.SeqId(s[itk+2])
                    if s[itk+3].lower().startswith("to"):
                        gr[-1]["resi_to"] = gemmi.SeqId(s[itk+4])
                    itk += 5
                else:
                    gr[-1]["resi"] = gemmi.SeqId(s[itk+1])
                    itk += 2
            elif s[itk].lower().startswith("atom"):
                gr[-1]["atom"] = s[itk+1]
                itk += 2
            elif s[itk].lower().startswith("alt"):
                gr[-1]["alt"] = s[itk+1]
                itk += 2
    elif (ntok > 4 and
          s[1].lower().startswith("grou") and
          s[2].lower().startswith("alts")):
        r["const"].append((s[3].lower().startswith("comp"), s[4:]))
    elif ntok > 1 and s[1].lower().startswith("refi"):
        if ntok > 3 and s[2].lower().startswith("ncyc"):
            r["ncycle"] = max(int(s[3]), r["ncycle"])
        elif r["ncycle"] == 0:
            r["ncycle"] = 1 # default

    return r
# read_occupancy_params()

def read_restr_params(l, r):
    s = l.split()

    def read_tors_params(itk):
        ret = {"flag": True}
        if s[itk].lower().startswith("none"): # remove all
            ret["flag"] = False
            itk += 1
        elif s[itk].lower().startswith("resi"):
            ret["residue"] = s[itk+1]
            itk += 2
        elif s[itk].lower().startswith("grou"):
            ret["group"] = s[itk+1] # group_name_tors_restr_o
            itk += 2
        elif s[itk].lower().startswith("link"):
            ret["link"] = s[itk+1] #link_name_tors_restr_o
            itk += 2
        else:
            pass # raise error?
        while itk < len(s):
            if s[itk].lower().startswith("name"):
                itk += 1
                ret["tors_name"] = s[itk] # RES_NAME_TORS_NAME_O
            elif s[itk].lower().startswith("valu"):
                itk += 1
                ret["tors_value"] = float(s[itk]) # RES_NAME_TORS_VALUE_O
            elif s[itk].lower().startswith("sigm"):
                itk += 1
                ret["tors_sigma"] = float(s[itk]) # RES_NAME_TORS_SIGMA_O
            elif s[itk].lower().startswith("peri"):
                itk += 1
                ret["tors_period"] = int(s[itk]) # RES_NAME_TORS_PERIOD_O
            itk += 1
        return ret, itk
    # read_tors_params()
    
    if not s[0].lower().startswith("rest"):
        return r
    if s[1].lower().startswith("excl"):
        # restr_params.f90 subroutine exclude_restraints
        #r["exclude"] = True
        pass
    elif s[1].lower().startswith("conf"): # not implemented in Refmac
        pass
    elif s[1].lower().startswith(("bp", "pair", "base")):
        if s[2].lower().startswith("dist"):
            r["plane_dist"] = True
        else:
            r["basepair"] = True
            # TODO read dnarna_params.txt
    elif s[1].lower().startswith("tors") and len(s) > 2:
        itk = 2
        if s[itk].lower().startswith("hydr"):
            itk += 1
            if s[itk].lower().startswith("incl"):
                itk += 1
                r["htors_restraint"] = True
                if s[itk].lower().startswith("all"): # this is servalcat default for now
                    r["htors_restraint_type"] = "all"
                elif s[itk].lower().startswith("sele"):
                    r["htors_restraint_type"] = "sele"
                    r["htors_restraint_list"] = s[itk+1:]
            else:
                itk += 1
                r["htors_restraint"] = False
        elif s[itk].lower().startswith("fbas"):
            pass # set user_basepair_file
        elif s[itk].lower().startswith("incl"):
            tmp, itk = read_tors_params(itk+1)
            r.setdefault("torsion_include", []).append(tmp)
        elif s[itk].lower().startswith("excl"):
            # Should warn if value/sigma/period given?
            tmp, itk = read_tors_params(itk+1)
            r.setdefault("torsion_exclude", []).append(tmp)
    elif s[1].lower().startswith("resi"):
        pass
    elif s[1].lower().startswith("chir"):
        pass
# read_restr_params()

def read_make_params(l, r):
    # TODO: hout,ribo,valu,spec,form,sdmi,segi
    s = l.split()
    assert s[0].lower().startswith("make")
    itk = 1
    ntok = len(s)
    # keyword, key, func, possible, default
    keys = (("hydr", "hydr", lambda x: x[0].lower(), set("aynf"), "a"),
            ("hout", "hout", lambda x: x[0].lower() in ("y", "p"), (True, False), True),
            ("chec", "check", lambda x: "0" if x.lower().startswith(("none", "0"))
             else ("n" if x.lower().startswith(("liga", "n"))
                   else ("y" if x.lower().startswith(("all", "y")) else None)),
             set("0ny"), None), # no default
            ("newl", "newligand", lambda x: x.lower().startswith(("c", "y", "noex")), (True, False), False),
            ("buil", "build", lambda x: x[0].lower(), set("ny"), "n"),
            ("pept", "pept", lambda x: x[0].lower(), set("yn"), "y"),
            ("link", "link", lambda x: x[0].lower(), set("ynd0"), "y"),
            ("suga", "sugar", lambda x: x[0].lower(), set("ynds"), "y"),
            #("conn", "conn", lambda x: x[0].lower(), set("nyd0"), "n"), # TODO read conn_tolerance? (make conn tole val)
            ("symm", "symm", lambda x: x[0].lower(), set("ync"), "y"),
            ("chai", "chain", lambda x: x[0].lower(), set("yn"), "y"),
            ("cisp", "cispept", lambda x: x[0].lower(), set("yn"), "y"),
            ("ss", "ss", lambda x: x[0].lower(), set("ydn"), "y"),
            ("exit", "exit", lambda x: x[0].lower() == "y", (True, False), True),
            )
    while itk < ntok:
        found = False
        for k, t, f, p, d in keys:
            if s[itk].lower().startswith(k):
                if itk + 1 < len(s):
                    r[t] = f(s[itk+1])
                    if r[t] not in p:
                        raise SystemExit("Invalid make instruction: {}".format(l))
                    itk += 2
                elif d is not None:
                    r[t] = d # set default
                    itk += 1
                else:
                    raise SystemExit("Invalid make instruction: {}".format(l))
                break
        else: # if no keywords match (can raise an error if all make keywords are implemented)
            itk += 1
    return r
# read_make_params()

def parse_line(l, ret):
    s = l.split()
    ntok = len(s)
    if ntok == 0: return
    if s[0].lower().startswith("exte"):
        ret.setdefault("exte", []).append(read_exte(s))
    elif s[0].lower().startswith("make"):
        read_make_params(l, ret.setdefault("make", {}))
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
        ret.setdefault("refi", {})
        itk = 1
        while itk < ntok:
            if s[itk].lower().startswith("type"):
                if itk+1 < ntok and s[itk+1].lower().startswith("unre"):
                    ret["refi"]["type"] = "unre"
                itk += 2
            else:
                itk += 1
    elif s[0].lower().startswith("dist"):
        try:
            ret["wbond"] = float(s[1])
        except:
            pass
        # TODO read sdex, excu, dele, dmxe, dmne
    elif s[0].lower().startswith("ridg"):
        read_ridge_params(l, ret.setdefault("ridge", {}))
    elif s[0].lower().startswith("occu"):
        read_occupancy_params(l, ret.setdefault("occu", {}))
    elif s[0].lower().startswith("rest"):
        read_restr_params(l, ret.setdefault("restr", {}))
    elif s[0].lower().startswith("angl") and ntok > 1:
        ret["wangle"] = float(s[1])
    elif s[0].lower().startswith("tors") and ntok > 1:
        ret["wtors"] = float(s[1])
    elif s[0].lower().startswith(("bfac", "temp", "bval")):
        pass # TODO
    elif s[0].lower().startswith("plan") and ntok > 1:
        ret["wplane"] = float(s[1])
    elif s[0].lower().startswith("chir") and ntok > 1:
        ret["wchir"] = float(s[1])
        # TODO read calp
    elif s[0].lower().startswith(("vdwr", "vand", "nonb")) and ntok > 1:
        itk = 1
        try:
            ret["wvdw"] = float(s[itk])
            itk += 1
        except ValueError:
            pass
        # TODO read maxr, over, sigm, incr, chan, vdwc, excl
# parse_line()

def get_lines(lines, depth=0):
    ret = []
    cont = ""
    for l in lines:
        if "!" in l: l = l[:l.index("!")]
        if "#" in l: l = l[:l.index("#")]
        l = l.strip()
        if not l: continue
        if l[0] == "@":
            f = l[1:]
            try:
                yield from get_lines(open(f).readlines(), depth+1)
            except RuntimeError:
                return
            continue
        if l.split()[-1] == "-":
            cont += l[:l.rfind("-")] + " "
            continue
        if cont:
            l = cont + l
            cont = ""
        if l.split()[0].lower().startswith("end"):
            # refmac stops reading keywords when "exit" is seen in stdin or a file
            # but won't do this from nested files
            if depth == 1:
                raise RuntimeError
            break
        yield l
# get_lines()
            
def update_params(ret, inputs):
    if not inputs:
        return
    for l in get_lines(inputs):
        parse_line(l, ret)
# update_keywords()

def parse_keywords(inputs):
    ret = {"make":{}, "ridge":{}, "refi":{}}
    update_params(ret, inputs)
    return ret
# parse_keywords()

if __name__ == "__main__":
    import sys
    import json
    print("waiting for input")
    ret = {} #{"make":{}, "ridge":{}, "refi":{}}
    for l in get_lines(sys.stdin):
        parse_line(l, ret)
    print()
    print("Parsed:")
    print(json.dumps(ret, indent=1))
