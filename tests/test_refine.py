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
import test_spa
from servalcat import utils
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
        self.assertLess(stats[-1]["data"]["summary"]["R1"], 0.26)
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

if __name__ == '__main__':
    unittest.main()

