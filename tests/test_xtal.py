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
import pipes
import hashlib
from servalcat import utils
from servalcat.xtal import sigmaa
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
        self.assertAlmostEqual(b, -8.48374, places=4)
    # test_scale()

    def test_sigmaa(self):
        mtzin = os.path.join(root, "5e5z", "5e5z.mtz.gz")
        pdbin = os.path.join(root, "5e5z", "5e5z.pdb.gz")
        args = sigmaa.parse_args(["--hklin", mtzin, "--model", pdbin, "--D_as_exp", "--S_as_exp",
                                  "--labin", "FP,SIGFP", "--nbins", "10"])
        hkldata = sigmaa.main(args)
        os.remove("sigmaa.log")
        os.remove("sigmaa.mtz")

        numpy.testing.assert_allclose(hkldata.binned_df.d_min,
                                      [5.0834, 3.6631, 3.0103, 2.6155, 2.344 , 2.1426, 1.9855, 1.8586,
                                       1.7532, 1.664 ],
                                      rtol=1e-4)
        numpy.testing.assert_allclose(hkldata.binned_df.D0,
                                      [0.844599, 0.983733, 1.022031, 0.965314, 0.996966, 1.009102,
                                       0.997227, 0.98949 , 0.937605, 0.928479],
                                      rtol=1e-5)
        numpy.testing.assert_allclose(hkldata.binned_df.D1,
                                      [2.974686e-01, 5.625804e-08, 2.500706e-10, 3.981426e-10,
                                       7.635458e-07, 8.245066e-08, 1.000051e+00, 9.999994e-01,
                                       9.999997e-01, 1.000001e+00],
                                      rtol=1e-5)
        numpy.testing.assert_allclose(hkldata.binned_df.S,
                                      [ 84.857332,  95.794274,  68.854357,  96.124265,  74.228905,
                                        110.075772, 116.446679,  95.912967,  81.069191,  43.255408],
                                      rtol=1e-5)

    # test_sigmaa()

    def test_refine_cx(self):
        mtzin = os.path.join(root, "biotin", "biotin_talos.mtz")
        pdbin = os.path.join(root, "biotin", "biotin_talos.pdb")
        sys.argv = ["", "refine_cx", "--model", pipes.quote(pdbin),
                    "--hklin", pipes.quote(mtzin), "--bref", "iso_then_aniso"]
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

