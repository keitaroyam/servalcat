"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
from servalcat import utils
b_to_u = utils.model.b_to_u

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

def read_exte_line(l):
    # using the same variable names as used in read_extra_restraints.f
    s = l.split()
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
                    try:
                        defs["scale_sigma_dist"] = float(s[itk+1]) # scale_sigma_loc
                        itk += 2
                    except ValueError:
                        pass
                    for k in ("angl", "tors", "chir", "plan", "dist", "inte"):
                        if s[itk+1].lower().startswith(k):
                            defs["scale_sigma_{}".format(k)]  = float(s[itk+2])
                            itk += 3
                elif s[itk].lower().startswith("sgmn"):
                    defs["sigma_min_loc"] = float(s[itk+1])
                    itk += 2
                elif s[itk].lower().startswith("sgmx"):
                    defs["sigma_max_loc"] = float(s[itk+1])
                    itk += 2
                else:
                    raise RuntimeError("Error==> EXTE keyword interpretation: {}".format(l))
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
                        logger.writeln("WARNING: wrong type is given. setting to 2.\n=> {}".format(l))
                        ret["restr"]["itype_in"] = 2
                    itk += 2
                elif s[itk].lower().startswith("symm"): # only for distance
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
                    logger.writeln("WARNING: unrecognised keyword: {}\n=> {}".format(s[itk], l))
                    itk += 1

        else:
            logger.writeln("WARNING: cannot parse: {}".format(l))
    return ret
# read_exte_line()

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

def read_make_params(l, r):
    # TODO: hout,ribo,valu,spec,form,sdmi,segi
    s = l.split()
    assert s[0].lower().startswith("make")
    itk = 1
    ntok = len(s)
    while itk < ntok:
        if s[itk].lower().startswith("hydr"): #default a
            r["hydr"] = s[itk+1][0].lower()
            if r["hydr"] not in "yanf":
                raise SystemExit("Invalid make instruction: {}".format(l))
            itk += 2
        elif s[itk].lower().startswith("hout"): # default n
            tmp = s[itk+1][0].lower()
            if tmp == "p": tmp = "y"
            r["hout"] = tmp == "y"
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
            if itk + 1 < len(s):
                r["exit"] = s[itk+1][0].lower() == "y"
                itk += 2
            else:
                r["exit"] = True
                itk += 1
        else:
            itk += 1
    return r
# read_make_params()

def parse_line(l, ret):
    s = l.split()
    ntok = len(s)
    if ntok == 0: return
    if s[0].lower().startswith("make"):
        read_make_params(l, ret["make"])
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
    elif s[0].lower().startswith("dist"):
        try:
            ret["wbond"] = float(s[1])
        except:
            pass
        # TODO read sdex, excu, dele, dmxe, dmne
    elif s[0].lower().startswith("ridg"):
        read_ridge_params(l, ret["ridge"])
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

def get_lines(lines):
    ret = []
    cont = ""
    for l in lines:
        if "!" in l: l = l[:l.index("!")]
        if "#" in l: l = l[:l.index("#")]
        l = l.strip()
        if not l: continue
        if l[0] == "@":
            f = l[1:]
            yield from get_lines(open(f).readlines())
            continue
        if l.split()[-1] == "-":
            cont += l[:l.rfind("-")] + " "
            continue
        if cont:
            l = cont + l
            cont = ""
        yield l
# get_lines()
            
def parse_keywords(inputs):
    ret = {"make":{}, "ridge":{}}
    for l in get_lines(inputs):
        if l.split()[0].lower().startswith("end"):
            break
        parse_line(l, ret)
    return ret
# parse_keywords()

