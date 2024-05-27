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
import test_spa
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
        stats = json.load(open("5e5z_refined_stats.json"))
        self.assertLess(stats[-1]["geom"]["summary"]["r.m.s.d."]["Bond distances, non H"], 0.01)
        
    def test_refine_spa(self):
        data = test_spa.data
        sys.argv = ["", "refine_spa_norefmac", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"],
                    "--resolution", "1.9", "--ncycle", "2",]
        main()
        self.assertTrue(os.path.isfile("refined_fsc.json"))
        self.assertTrue(os.path.isfile("refined.mmcif"))
        self.assertTrue(os.path.isfile("refined_diffmap.mtz"))
        self.assertTrue(os.path.isfile("refined_expanded.pdb"))
        
        stats = json.load(open("refined_stats.json"))
        self.assertGreater(stats[-1]["data"]["summary"]["FSCaverage"], 0.66)
        
if __name__ == '__main__':
    unittest.main()

