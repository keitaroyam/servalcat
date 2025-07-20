"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import gemmi
import numpy
from servalcat.utils import logger
from servalcat import ext

"""import line_profiler
profile = line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)
@profile"""
def read_external_restraints(params, st, geom):
    # default or current values
    defs = dict(symall_block=False, exclude_self_block=False, type_default=2, alpha_default=1.,
                ext_verbose=False, scale_sigma_dist=1., scale_sigma_angl=1., scale_sigma_tors=1.,
                scale_sigma_chir=1., scale_sigma_plan=1., scale_sigma_inte=1.,
                sigma_min_loc=0., sigma_max_loc=100., ignore_undefined=False, ignore_hydrogens=True,
                dist_max_external=numpy.inf, dist_min_external=-numpy.inf, use_atoms="a", prefix_ch=" ")
    #exte = gemmi.ExternalRestraints(st)
    extypes = dict(dist=ext.Geometry.Bond,
                   angl=ext.Geometry.Angle,
                   chir=ext.Geometry.Chirality,
                   tors=ext.Geometry.Torsion,
                   plan=ext.Geometry.Plane,
                   inte=ext.Geometry.Interval,
                   harm=ext.Geometry.Harmonic,
                   spec=ext.Geometry.Special,
                   stac=ext.Geometry.Stacking)
    exlists = dict(dist=geom.bonds, angl=geom.angles, tors=geom.torsions,
                   chir=geom.chirs, plan=geom.planes, inte=geom.intervals,
                   stac=geom.stackings, harm=geom.harmonics, spec=geom.specials)
    num_org = {x: len(exlists[x]) for x in exlists}

    # XXX There may be duplication (same chain, resi, name, and alt) - we should give error?
    lookup = {(cra.chain.name, cra.residue.seqid.num, cra.residue.seqid.icode,
               cra.atom.name, cra.atom.altloc) : cra.atom for cra in st[0].all()}

    # TODO main chain / side chain filtering, hydrogen, dist_max_external/dist_min_external
    for r in params:
        if not r: continue
        defs.update(r["defaults"])
        if "rest_type" not in r: continue
        if r["rest_type"] not in extypes:
            logger.writeln("Warning: unknown external restraint type: {}".format(r["rest_type"]))
            continue

        atoms = []
        skip = False
        for i, spec in enumerate(r["restr"].get("specs", [])):
            if r["rest_type"] == "stac":
                atoms.append([])
            if "ifirst" in spec:
                for chain in st[0]:
                    if chain.name != spec["chain"]: continue
                    for res in chain:
                        if spec["ifirst"] is not None and res.seqid.num < spec["ifirst"]: continue
                        if spec["ilast"] is not None and res.seqid.num > spec["ilast"]: continue
                        atoms.extend([a for a in res if spec.get("atom", "*") == "*" or a.name == spec["atom"]])
            else:
                for name in spec["names"]: # only same altloc allowed?
                    key = (spec["chain"], spec["resi"], spec.get("icode", " "),
                           name, spec.get("altloc", "\0"))
                    atom = lookup.get(key)
                    if atom is None:
                        if defs["ignore_undefined"]:
                            logger.writeln("Warning: atom not found: {}".format(key))
                            skip = True
                            continue
                        raise RuntimeError("Atom not found: {}".format(key))
                    if r["rest_type"] == "stac":
                        atoms[i].append(atom)
                    else:
                        atoms.append(atom)
        if skip or not atoms:
            continue
        if r["rest_type"] in ("spec", "harm"):
            if r["restr"]["rectype"] == "auto":
                assert r["rest_type"] == "spec"
                atoms = [cra.atom for cra in st[0].all()]
            for atom in atoms:
                ex = extypes[r["rest_type"]](atom)
                if r["rest_type"] == "spec":
                    # TODO check if it is on special position. using r["restr"]["toler"]
                    ex.sigma_t = r["restr"]["sigma_t"]
                    ex.sigma_u =r["restr"]["sigma_u"]
                    ex.u_val_incl = r["restr"]["u_val_incl"]
                    # ex.trans_t =
                    # ex.mat_u = 
                else:
                    ex.sigma = r["restr"]["sigma_t"]
                exlists[r["rest_type"]].append(ex)
            continue
        elif r["rest_type"] == "plan":
            ex = extypes[r["rest_type"]](atoms)
        else:
            ex = extypes[r["rest_type"]](*atoms)
        if r["rest_type"] in ("dist", "angl", "chir", "tors"):
            value = r["restr"]["value"]
            sigma = r["restr"]["sigma_value"] / defs["scale_sigma_{}".format(r["rest_type"])]
            if r["rest_type"] == "chir":
                ex.value = value
                ex.sigma = sigma
            else:
                if r["rest_type"] == "dist":
                    sigma = min(max(sigma, defs["sigma_min_loc"]), defs["sigma_max_loc"])
                    vals = (value, sigma, value, sigma) # nucleus
                elif r["rest_type"] == "tors":
                    vals = (value, sigma, 1) # period. # Refmac does not seem to read it from instruction
                else:
                    vals = (value, sigma)
                ex.values.append(extypes[r["rest_type"]].Value(*vals))
        
        if r["rest_type"] == "dist":
            if not (defs["dist_min_external"] < r["restr"]["value"] < defs["dist_max_external"]):
                continue
            ex.alpha = r["restr"].get("alpha_in", defs["alpha_default"])
            ex.type = r["restr"].get("itype_in", defs["type_default"])
            symm1 = any([spec.get("symm") for spec in r["restr"]["specs"]]) # is it the intention?
            if r["restr"].get("symm_in", defs["symall_block"]) or symm1:
                asu = gemmi.Asu.Different if defs["exclude_self_block"] else gemmi.Asu.Any
                ex.set_image(st.cell, asu)
            #print("dist=", ex.alpha, ex.type, ex.values[-1].value, ex.values[-1].sigma, ex.sym_idx, ex.pbc_shift, ex.atoms)
        elif r["rest_type"] == "angl":
            if any(spec.get("symm") for spec in r["restr"]["specs"]):
                asus = [gemmi.Asu.Different if r["restr"]["specs"][i].get("symm") else gemmi.Asu.Same
                        for i in range(3)]
                if atoms[0].serial > atoms[2].serial:
                    asus = asus[::-1]
                ex.set_images(st.cell, asus[0], asus[2])
            #print("angl=", ex.values[-1].value, ex.values[-1].sigma, ex.atoms)
        elif r["rest_type"] == "tors":
            pass
            #print("tors=", ex.values[-1].value, ex.values[-1].sigma, ex.atoms)
        elif r["rest_type"] == "chir":
            #print("chir=", ex.value, ex.sigma, ex.atoms)
            ex.sign = gemmi.ChiralityType.Positive if ex.value > 0 else gemmi.ChiralityType.Negative
            ex.value = abs(ex.value)
        elif r["rest_type"] == "plan":
            ex.sigma = r["restr"]["sigma_value"] / defs["scale_sigma_{}".format(r["rest_type"])]
            #print("plan=", ex.sigma, ex.atoms)
        elif r["rest_type"] == "inte":
            dmin, dmax = r["restr"].get("dmin"), r["restr"].get("dmax")
            smin, smax = r["restr"].get("smin"), r["restr"].get("smax")
            if (smin,smax).count(None) == 2:
                smin = smax = 0.05
            else:
                if smin is None: smin = smax
                if smax is None: smax = smin
                smin /= defs["scale_sigma_inte"]
                smax /= defs["scale_sigma_inte"]
            if (dmin,dmax).count(None) == 1:
                if dmin is None: dmin = dmax
                if dmax is None: dmax = dmin
            ex.dmin = dmin
            ex.dmax = dmax
            ex.smin = smin
            ex.smax = smax
            symm1 = any(spec.get("symm") for spec in r["restr"]["specs"]) # not tested
            if r["restr"].get("symm_in", defs["symall_block"]) or symm1:
                asu = gemmi.Asu.Different if defs["exclude_self_block"] else gemmi.Asu.Any
                ex.set_image(st.cell, asu)
            #print("inte=", ex.dmin, ex.dmax, ex.smin, ex.smax, ex.atoms)
        elif r["rest_type"] == "stac":
            ex.dist = r["restr"]["dist_id"]
            ex.sd_dist = r["restr"]["dist_sd"]
            ex.angle = r["restr"].get("angle_id", 0.)
            ex.sd_angle = r["restr"]["angle_sd"]
            #print("stac=", ex.dist, ex.sd_dist, ex.angle, ex.sd_angle, ex.planes)
            
        exlists[r["rest_type"]].append(ex)

    logger.writeln("External restraints from Refmac instructions")
    labs = dict(dist="distances", angl="angles", tors="torsions",
                chir="chirals", plan="planes", inte="intervals",
                stac="stackings", harm="harmonics", spec="special positions")
    for lab in labs:
        logger.writeln(" Number of {:18s} : {}".format(labs[lab], len(exlists[lab]) - num_org[lab]))
    logger.writeln("")
# read_external_restraints()
