"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
import unittest
import numpy
import pandas
import json
import gemmi
import os
import shutil
import sys
import tempfile
import hashlib
from servalcat import utils
from servalcat.xtal import sigmaa
from servalcat.xtal import french_wilson
from servalcat.__main__ import main

root = os.path.abspath(os.path.dirname(__file__))

class XtalTests(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        print("In", self.wd)
    # setUp()
    
    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)
    # tearDown()

    def test_scale(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        st = utils.fileio.read_structure(pdbin)
        mtz = gemmi.read_mtz_file(mtzin)
        hkldata = utils.hkl.hkldata_from_asu_data(mtz.get_float("I"), "I")

        hkldata.df["FC"] = utils.model.calc_fc_fft(st, mtz.resolution_high(), "xray", mott_bethe=False,
                                                   miller_array=hkldata.miller_array())
        hkldata.df["IC"] = numpy.abs(hkldata.df["FC"].to_numpy())**2
        k, b = hkldata.scale_k_and_b("I", "IC")
        self.assertAlmostEqual(k, 0.00665, places=5)
        self.assertAlmostEqual(b, -8.6132, places=3)
    # test_scale()

    def test_sigmaa(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_trans", "exp", "--S_trans", "exp",
                                  "--labin", "FP,SIGFP", "--nbins", "10", "--nbins_ml", "10", "--source", "xray"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.log")
        os.remove("sigmaa.mtz")

        numpy.testing.assert_allclose(hkldata.binned_df["ml"].d_min,
                                      [5.0834, 3.6631, 3.0103, 2.6155, 2.344 , 2.1426, 1.9855, 1.8586,
                                       1.7532, 1.664 ],
                                      rtol=1e-4)
        numpy.testing.assert_allclose(hkldata.binned_df["ml"].D0,
                                      [0.829663, 0.993061, 1.033881, 0.982557, 1.016779, 1.035846,
                                       1.025311, 1.023788, 0.970901, 0.974236],
                                      rtol=1e-2)
        #numpy.testing.assert_allclose(hkldata.binned_df["ml"].D1,
        #                              [2.974164e-01, 5.624910e-08, 2.500540e-10, 3.980775e-10,
        #                               7.638667e-07, 8.243909e-08, 1.000051e+00, 9.999994e-01,
        #                               9.999997e-01, 1.000001e+00],
        #                              rtol=1e-5)
        numpy.testing.assert_allclose(hkldata.binned_df["ml"].S,
                                      [90.147006,  97.960666,  67.314535,  97.67571 ,  72.546305,
                                       109.631114, 119.247475,  98.844853,  78.992354,  41.845315],
                                      rtol=1e-2)

    # test_sigmaa()
    
    def test_sigmaa_int(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_trans", "splus", "--S_trans", "splus",
                                  "--labin", "I,SIGI", "--nbins", "10", "--nbins_ml", "10", "--source", "xray"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.mtz")
        numpy.testing.assert_allclose(hkldata.binned_df["ml"].D0,
                                      [0.829305, 0.98676 , 1.024622, 0.967299, 0.998059,
                                       1.014157, 1.000922, 0.995466, 0.93671 , 0.932352],
                                      rtol=1e-2)
        numpy.testing.assert_allclose(hkldata.binned_df["ml"].S,
                                      [89.637282,  97.784362,  68.782025,  99.382283,  74.905775,
                                       112.326708, 126.651614, 106.853499,  83.032016,  42.909208],
                                      rtol=1e-2)

    def test_fw(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        args = french_wilson.parse_args(["--hklin", mtzin, "--labin", "I,SIGI"])
        B_aniso, hkldata = french_wilson.main(args)
        os.remove("5e5z_fw.mtz")
        numpy.testing.assert_allclose(B_aniso.elements_pdb(),
                                      [2.640011, 1.679485, -4.319497, 0. ,-1.072883, 0.],
                                      rtol=1e-3)                                       

    @unittest.skipUnless(utils.refmac.check_version(), "refmac unavailable")
    def test_refine_cx(self):
        mtzin = os.path.join(root, "biotin", "biotin_talos.mtz")
        pdbin = os.path.join(root, "biotin", "biotin_talos.pdb")
        sys.argv = ["", "refine_cx", "--model", pdbin,
                    "--hklin", mtzin, "--bref", "iso_then_aniso"]
        main()
        self.assertTrue(os.path.isfile("refined_2_aniso.mtz"))
        r_factor = None
        with open("refined_2_aniso.log") as ifs:
            for l in ifs:
                if l.startswith("           R factor"):
                    r_factor = float(l.split()[-1])
        self.assertLess(r_factor, 0.185)
    # test_refine_cx()
    
# class XtalTests

if __name__ == '__main__':
    unittest.main()

