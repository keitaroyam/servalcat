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
from servalcat import command_line

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
        self.assertAlmostEqual(k, 0.00667, places=5)
        self.assertAlmostEqual(b, -8.48166, places=3)
    # test_scale()

    def test_sigmaa(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_trans", "exp", "--S_trans", "exp",
                                  "--labin", "FP,SIGFP", "--nbins", "10", "--source", "xray"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.log")
        os.remove("sigmaa.mtz")

        numpy.testing.assert_allclose(hkldata.binned_df.d_min,
                                      [5.0834, 3.6631, 3.0103, 2.6155, 2.344 , 2.1426, 1.9855, 1.8586,
                                       1.7532, 1.664 ],
                                      rtol=1e-4)
        numpy.testing.assert_allclose(hkldata.binned_df.D0,
                                      [0.844594, 0.983737, 1.022041, 0.965311, 0.996949, 1.009086,
                                       0.997229, 0.989502, 0.937616, 0.928479],
                                      rtol=1e-2)
        #numpy.testing.assert_allclose(hkldata.binned_df.D1,
        #                              [2.974164e-01, 5.624910e-08, 2.500540e-10, 3.980775e-10,
        #                               7.638667e-07, 8.243909e-08, 1.000051e+00, 9.999994e-01,
        #                               9.999997e-01, 1.000001e+00],
        #                              rtol=1e-5)
        numpy.testing.assert_allclose(hkldata.binned_df.S,
                                      [84.849295, 95.793834, 68.855376, 96.128495, 74.232567,
                                       110.074536, 116.444811, 95.912754, 81.073520, 43.255423],
                                      rtol=1e-2)

    # test_sigmaa()
    
    def test_sigmaa_int(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_trans", "splus", "--S_trans", "splus",
                                  "--labin", "I,SIGI", "--nbins", "10", "--source", "xray"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.mtz")
        numpy.testing.assert_allclose(hkldata.binned_df.D0,
                                      [0.845341, 0.982548, 1.021152, 0.960991, 0.991905,
                                       1.004682, 0.991878, 0.983793, 0.927995, 0.91412],
                                      rtol=1e-2)
        numpy.testing.assert_allclose(hkldata.binned_df.S,
                                      [85.040581, 96.759372, 70.447608, 97.630793, 76.895628,
                                       112.826406, 124.749979, 103.492015, 84.482203, 44.150007],
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
        command_line.main()
        self.assertTrue(os.path.isfile("refined_2_aniso.mtz"))
        r_factor = None
        for l in open("refined_2_aniso.log"):
            if l.startswith("           R factor"):
                r_factor = float(l.split()[-1])
        self.assertLess(r_factor, 0.185)
    # test_refine_cx()
    
# class XtalTests

if __name__ == '__main__':
    unittest.main()

