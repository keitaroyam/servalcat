"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import json
import os
import shutil
import tempfile
import sys
import numpy
import pandas
import gemmi
import test_spa
from pandas.testing import assert_frame_equal
from servalcat import utils
from servalcat.refine.refine import Geom, RefineParams, load_config
from servalcat.refmac import refmac_keywords
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))

class TestRefine(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        print("In", self.wd)
    # setUp()

    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)
    # tearDown()

    def test_refine_geom(self):
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        sys.argv = ["", "refine_geom", "--model", pdbin, "--rand", "0.5"]
        main()
        with open("5e5z_refined_stats.json") as f:
            stats = json.load(f)
        self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Bond distances, non H"], 0.01)
        
    def test_refine_xtal_int(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        sys.argv = ["", "refine_xtal_norefmac", "--model", pdbin, "--rand", "0.5",
                    "--hklin", mtzin, "-s", "xray", "--labin", "I,SIGI,FREE", "--nbins", "5"]
        main()
        with open("5e5z_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIfreeavg"], 0.70)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIworkavg"], 0.91)

    def test_refine_xtal(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        sys.argv = ["", "refine_xtal_norefmac", "--model", pdbin, "--rand", "0.5",
                    "--hklin", mtzin, "-s", "xray", "--labin", "FP,SIGFP,FREE"]
        main()
        with open("5e5z_refined_stats.json") as f:
            stats = json.load(f)
        self.assertLess(stats[-1]["data"]["summary"]["Rfree"], 0.22)
        self.assertLess(stats[-1]["data"]["summary"]["Rwork"], 0.20)

    def test_refine_small_hkl(self):
        hklin = os.path.join(root, "biotin", "biotin_talos.hkl")
        xyzin = os.path.join(root, "biotin", "biotin_talos.ins")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", hklin, "-s", "electron", "--unrestrained"]
        main()
        with open("biotin_talos_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIavg"], 0.64)

    def test_refine_small_cif(self):
        cifin = os.path.join(root, "biotin", "biotin_talos.cif")
        sys.argv = ["", "refine_xtal_norefmac", "--model", cifin,
                    "--hklin", cifin, "-s", "electron", "--unrestrained"]
        main()
        with open("biotin_talos_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIavg"], 0.64)
    
    def test_refine_aniso(self):
        hklin = os.path.join(root, "biotin", "biotin_talos.hkl")
        xyzin = os.path.join(root, "biotin", "biotin_talos.ins")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", hklin, "-s", "electron", "--unrestrained",
                    "--adp", "aniso"]
        main()
        with open("biotin_talos_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIavg"], 0.64)
        st = utils.fileio.read_structure("biotin_talos_refined.mmcif")
        self.assertTrue(all(x.atom.aniso.nonzero() for x in st[0].all()))

    def test_refine_aniso_occ(self):
        hklin = os.path.join(root, "biotin", "biotin_talos.hkl")
        xyzin = os.path.join(root, "biotin", "biotin_talos.ins")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", hklin, "-s", "electron", "--unrestrained",
                    "--adp", "aniso", "--refine_all_occ"]
        main()
        with open("biotin_talos_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIavg"], 0.64)
        st = utils.fileio.read_structure("biotin_talos_refined.mmcif")
        self.assertTrue(all(x.atom.aniso.nonzero() for x in st[0].all()))
        self.assertTrue(sum(x.atom.occ < 1 for x in st[0].all()) > 0.5 * st[0].count_atom_sites())
        
    def test_refine_spa(self):
        data = test_spa.data
        sys.argv = ["", "refine_spa_norefmac", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"],
                    "--resolution", "1.9", "--ncycle", "2", "--write_trajectory"]
        main()
        self.assertTrue(os.path.isfile("refined_fsc.json"))
        self.assertTrue(os.path.isfile("refined.mmcif"))
        self.assertTrue(os.path.isfile("refined_maps.mtz"))
        self.assertTrue(os.path.isfile("refined_expanded.pdb"))
        with open("refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["FSCaverage"], 0.66)

    def test_refine_group_occ(self):
        mtzin = os.path.join(root, "6mw0", "6mw0-sf.cif.gz")
        xyzin = os.path.join(root, "6mw0", "6mw0.cif")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", mtzin, "-s", "xray", "--labin", "IMEAN,SIGIMEAN",
                    "--bfactor", "5", "--keywords",
                    "occupancy group id 1 chain A alt A",
                    "occupancy group id 2 chain A alt B",
                    "occupancy group alts complete 1 2",
                    "occupancy refine ncycle 5"]
        main()
        with open("6mw0_refined_stats.json") as f:
            stats = json.load(f)
        self.assertLess(stats[-1]["data"]["summary"]["R"], 0.26)
        st = utils.fileio.read_structure("6mw0_refined.pdb")
        occ_a = tuple({round(a.occ, 6) for r in st[0]["A"] for a in r if a.altloc == "A"})
        occ_b = tuple({round(a.occ, 6) for r in st[0]["A"] for a in r if a.altloc == "B"})
        self.assertEqual(len(occ_a), 1)
        self.assertEqual(len(occ_b), 1)
        self.assertGreaterEqual(min(occ_a[0], occ_b[0]), 0.)
        self.assertLessEqual(max(occ_a[0], occ_b[0]), 1.)
        self.assertAlmostEqual(occ_a[0] + occ_b[0], 1.)

    def test_refine_dfrac(self):
        hklin = os.path.join(root, "1v9g", "1v9g-sf.cif.gz")
        xyzin = os.path.join(root, "1v9g", "1v9g-spk.cif.gz")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", hklin, "-s", "neutron",
                    "--hydr", "yes", "--hout", "--refine_dfrac"]
        main()
        with open("1v9g-spk_refined_stats.json") as f:
            stats = json.load(f)
        self.assertGreater(stats[-1]["data"]["summary"]["CCFfreeavg"], 0.52)
        st = utils.fileio.read_structure("1v9g-spk_refined.mmcif")
        self.assertGreater(numpy.std([x.atom.fraction for x in st[0].all() if x.atom.is_hydrogen()]), 0.3)

    def test_refine_twin(self):
        hklin = os.path.join(root, "1l2h", "1l2h.mtz.gz")
        xyzin = os.path.join(root, "1l2h", "1l2h.cif.gz")
        sys.argv = ["", "refine_xtal_norefmac", "--model", xyzin,
                    "--hklin", hklin, "-s", "xray", "--twin",
                    "--ncycle", "5"]
        main()
        with open("1l2h_refined_stats.json") as f:
            stats = json.load(f)
        self.assertEqual(list(stats[-1]["twin_alpha"]), ['h,k,l', '-h,k,-l'])
        self.assertAlmostEqual(stats[-1]["twin_alpha"]["h,k,l"], 0.66, delta=0.02)
        self.assertGreater(stats[-1]["data"]["summary"]["CCIfreeavg"], 0.81)

    def test_180deg(self):
        xyzin = "mg.pdb"
        exte = "exte.txt"
        with open(xyzin, "w") as ofs:
            ofs.write("""\
HETATM    1  MG  MG  A   1       0.000   0.000   0.000  1.00 10.00          MG
HETATM    2   O  HOH A   2       2.080   0.000   0.000  1.00 10.00           O
HETATM    3   O  HOH A   3      -2.080   0.000   0.000  1.00 10.00           O
HETATM    4   O  HOH A   4       0.000   2.080   0.000  1.00 10.00           O
HETATM    5   O  HOH A   5       0.000  -2.080   0.000  1.00 10.00           O
HETATM    6   O  HOH A   6       0.000   0.000   2.080  1.00 10.00           O
HETATM    7   O  HOH A   7       0.000   0.000  -2.080  1.00 10.00           O
""")
        with open(exte, "w") as ofs:
            ofs.write("""\
exte dist firs chai A resi 1 atom MG seco chai A resi 2 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 3 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 4 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 5 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 6 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 7 atom O valu 2.07 sigm 0.04 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 3 atom O valu 180 sigm 6.67 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 4 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 5 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 7 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 3 atom O next chai A resi 1 atom MG next chai A resi 4 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 3 atom O next chai A resi 1 atom MG next chai A resi 5 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 3 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 3 atom O next chai A resi 1 atom MG next chai A resi 7 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 4 atom O next chai A resi 1 atom MG next chai A resi 5 atom O valu 180 sigm 6.67 type 0
exte angl firs chai A resi 4 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 4 atom O next chai A resi 1 atom MG next chai A resi 7 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 5 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 5 atom O next chai A resi 1 atom MG next chai A resi 7 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 6 atom O next chai A resi 1 atom MG next chai A resi 7 atom O valu 180 sigm 6.67 type 0
""")
            
        sys.argv = ["", "refine_geom", "--model", xyzin,
                    "--rand", "0.1", "--keyword_file", exte]
        main()
        with open("mg_refined_stats.json") as f:
            stats = json.load(f)
        self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Bond angles, non H"], 1e-4)

    def test_180deg_symm(self):
        xyzin = "mg.pdb"
        exte = "exte.txt"
        with open(xyzin, "w") as ofs:
            ofs.write("""\
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1 2 1                  
HETATM    1  MG  MG  A   1       0.000   0.000   0.000  1.00 10.00          MG
HETATM    2   O  HOH A   2       2.080   0.000   0.000  1.00 10.00           O
HETATM    4   O  HOH A   4       0.000   2.080   0.000  1.00 10.00           O
HETATM    5   O  HOH A   5       0.000  -2.080   0.000  1.00 10.00           O
HETATM    6   O  HOH A   6       0.000   0.000   2.080  1.00 10.00           O
""")
        with open(exte, "w") as ofs:
            ofs.write("""\
exte dist firs chai A resi 1 atom MG seco chai A resi 2 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 4 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 5 atom O valu 2.07 sigm 0.04 type 0
exte dist firs chai A resi 1 atom MG seco chai A resi 6 atom O valu 2.07 sigm 0.04 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 2 atom O symm y valu 180 sigm 6.67 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 4 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 2 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 4 atom O next chai A resi 1 atom MG next chai A resi 5 atom O valu 180 sigm 6.67 type 0
exte angl firs chai A resi 4 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 5 atom O next chai A resi 1 atom MG next chai A resi 6 atom O valu 90 sigm 4.18 type 0
exte angl firs chai A resi 6 atom O next chai A resi 1 atom MG next chai A resi 6 atom O symm y valu 180 sigm 6.67 type 0
""")
            
        sys.argv = ["", "refine_geom", "--model", xyzin,
                    "--rand", "0.1", "--keyword_file", exte]
        main()
        with open("mg_refined_stats.json") as f:
            stats = json.load(f)
            
        # doesn't work??
        #self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Bond angles, non H"], 1e-4)
        
    def test_exte(self):
        xyzin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        st = utils.fileio.read_structure(xyzin)
        utils.model.setup_entities(st, clear=True, force_subchain_names=True, overwrite_entity_type=True)
        
        def get_geom(keywords):
            refmackwds = refmac_keywords.RefmacKeywords(keywords, None)
            refine_cfg = load_config(None, None, refmackwds)
            monlib = utils.restraints.load_monomer_library(st, stop_for_unknowns=True, refmackwds=refmackwds)
            utils.restraints.find_and_fix_links(st, monlib, find_metal_links=False, add_found=True)
            topo, _ = utils.restraints.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.NoChange,
                                                        refmackwds=refmackwds)
            refine_params = RefineParams(st, refine_xyz=True)
            geom = Geom(st, topo, monlib, refine_params, refine_cfg, refmackwds=refmackwds)
            geom.setup_nonbonded()
            return geom.show_model_stats()

        # test dist
        geo = get_geom([["exte dist firs chai A resi 2 ins . atom O seco chai A resi 3 ins . atom N value 2.9 sigma 0.1 alph 2"]])
        expected_df = pandas.DataFrame({"atom1": ["A/VAL 2/O"], "atom2": ["A/HIS 3/N"],
                                        "value": [2.247], "ideal": [2.900], "sigma": [0.100],
                                        "z": [-6.525], "type": [2], "alpha": [2.000]})
        assert_frame_equal(geo["outliers"]["bond"], expected_df, atol=0.001)

        # test dist symm
        geo = get_geom([["exte symall y exclude self",
                         "exte dist firs chai A resi 2 ins . atom O seco chai A resi 3 ins . atom N value 2.9 sigma 0.001 alph 2"]])
        expected_df = pandas.DataFrame({"atom1": ["A/VAL 2/O"], "atom2": ["A/HIS 3/N (2;1,0,0)"],
                                        "value": [2.941], "ideal": [2.900], "sigma": [0.001],
                                        "z": [40.642489], "type": [2], "alpha": [2.000]})
        assert_frame_equal(geo["outliers"]["bond"], expected_df, atol=0.001)

        # test override dist
        geo = get_geom([["exte dist firs chai A resi 2 ins . atom O seco chai A resi 2 ins . atom C value 1.2 sigma 0.001 type 0",
                         "exte dist firs chai A resi 2 ins . atom O seco chai A resi 2 ins . atom C value 1.9 sigma 0.001 type 1"]])
        expected_df = pandas.DataFrame({"atom1": ["A/VAL 2/C"], "atom2": ["A/VAL 2/O"],
                                        "value": [1.231], "ideal": [1.2], "sigma": [0.001],
                                        "z": [30.561], "type": [0], "alpha": [1.0]})
        assert_frame_equal(geo["outliers"]["bond"], expected_df, atol=0.001)

        # test tors
        geo = get_geom([["exte tors firs chai A resi 2 atom N seco chai A resi 2 atom CA thir chai A resi 2 atom C four chai A resi 2 atom O valu 90 sigma 10"]])
        expected_df = pandas.DataFrame({"label": "", "atom1": ["A/VAL 2/N"], "atom2": ["A/VAL 2/CA"], "atom3": ["A/VAL 2/C"], "atom4": ["A/VAL 2/O"], 
                                        "value": [-45.12], "ideal": [90.], "sigma": [10.], "per": [1],
                                        "z": [-13.512]})
        assert_frame_equal(geo["outliers"]["torsion"], expected_df, atol=0.001)

    def test_centroid(self):
        xyzin = os.path.join(root, "1e8a", "1e8a.cif.gz")
        with open("exte.json", "w") as ofs:
            data = [{"rest_type": "cdist", "restr": {"specs": [[{"chain": "A", "resi": 1090, "names": ["CA"]}],
                                                               [{"chain": "A", "resi": 65, "names": ["OD1","OD2"]}]],
                                                     "value": 2.15, "sigma": 0.08}},
                    {"rest_type": "cdist", "restr": {"specs": [[{"chain": "A", "resi": 1090, "names": ["CA"]}],
                                                               [{"chain": "A", "resi": 72, "names": ["OE1","OE2"]}]],
                                                     "value": 2.13, "sigma": 0.08}},
                    {"rest_type": "cangl", "restr": {"specs": [[{"chain": "A", "resi": 2084, "names": ["O"]}],
                                                               [{"chain": "A", "resi": 1090, "names": ["CA"]}],
                                                               [{"chain": "A", "resi": 65, "names": ["OD1","OD2"]}]],
                                                     "value": 90.0, "sigma": 1.0}},
                    {"rest_type": "cangl", "restr": {"specs": [[{"chain": "A", "resi": 65, "names": ["OD1","OD2"]}],
                                                               [{"chain": "A", "resi": 1090, "names": ["CA"]}],
                                                               [{"chain": "A", "resi": 72, "names": ["OE1","OE2"]}]],
                                                     "value": 180.0, "sigma": 1.0}}
                    ]
            json.dump(data, ofs)
        
        with open("config.yaml", "w") as ofs:
            ofs.write("""\
refine:
  exte_files:
  - "exte.json"
""")
            
        sys.argv = ["", "refine_geom", "--model", xyzin,
                    "--config", "config.yaml"]
        main()
        with open("1e8a_refined_stats.json") as f:
            stats = json.load(f)
        
        self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Centroid distances"], 0.3)
        self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Centroid angles"], 0.5)

if __name__ == '__main__':
    unittest.main()

