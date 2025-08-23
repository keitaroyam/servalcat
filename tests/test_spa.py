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
from servalcat.__main__ import main

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

root = os.path.abspath(os.path.dirname(__file__))
data = {}

def download():
    wd = os.path.join(root, "7dy0")
    if not os.path.exists(wd):
        os.mkdir(wd)

    url_md5 = (('half1', 'https://files.wwpdb.org/pub/emdb/structures/EMD-30913/other/emd_30913_half_map_1.map.gz', '0bf81c0057d3017972e7cf53e8a8df77'),
               ('half2', 'https://files.wwpdb.org/pub/emdb/structures/EMD-30913/other/emd_30913_half_map_2.map.gz', '5bba9c32a95d6675fe0387342af5a439'),
               ('mask', 'https://files.wwpdb.org/pub/emdb/structures/EMD-30913/masks/emd_30913_msk_1.map', 'e69403cec5be349b097c8080ce5bb1fd'),
               ('pdb', 'https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/dy/pdb7dy0.ent.gz', '7ddca7f90c2ebc4b73f2c14136cbda0c'),
               ('mmcif', 'https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/dy/7dy0.cif.gz', 'a0221f46f5e21bb6f53d5f8751d22444'))

    for name, url, md5 in url_md5:
        dst = os.path.join(root, "7dy0", os.path.basename(url))
        if not os.path.exists(dst):
            print("downloading {}".format(url))
            urlretrieve(url, dst)

        with open(dst, "rb") as f:
            print("testing md5sum of {}".format(dst))
            md5f = hashlib.md5(f.read()).hexdigest()
        assert md5 == md5f
        data[name] = dst

download()

def load_halfmaps():
    return [utils.fileio.read_ccp4_map(data[f]) for f in ("half1", "half2")]
def load_mask():
    return utils.fileio.read_ccp4_map(data["mask"])[0]
def load_pdb():
    return utils.fileio.read_structure(data["pdb"])
def load_mmcif():
    return utils.fileio.read_structure(data["mmcif"])

class TestFunctions(unittest.TestCase):
    def test_h_add(self):
        st = load_mmcif()
        self.assertFalse(st[0].has_hydrogen())
        monlib = utils.restraints.load_monomer_library(st)
        utils.restraints.add_hydrogens(st, monlib, "elec")
        self.assertEqual(st[0].count_atom_sites(gemmi.Selection('[H,D]')), 841)
        bad_h = [numpy.allclose(cra.atom.pos.tolist(), [0,0,0]) for cra in st[0].all()]
        self.assertEqual(sum(bad_h), 0)
    # test_h_add()

    def test_halfmaps_stats(self):
        maps = load_halfmaps()
        mask = load_mask()
        d_min = utils.maps.nyquist_resolution(maps[0][0])
        self.assertAlmostEqual(d_min, 1.6, places=3)
        hkldata = utils.maps.mask_and_fft_maps(maps, d_min, mask)
        hkldata.setup_relion_binning("ml")
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        numpy.testing.assert_allclose(hkldata.binned_df["ml"].var_signal,
                                      [3.01280911e+04, 2.73253902e+03, 1.12831493e+03, 7.78811837e+02,
                                       8.62027816e+02, 6.75563283e+02, 5.04824473e+02, 3.68568664e+02,
                                       3.08512467e+02, 2.24858349e+02, 2.01293139e+02, 2.07807437e+02,
                                       1.95582559e+02, 1.88518597e+02, 2.58906187e+02, 3.79117006e+02,
                                       4.89705517e+02, 5.22415086e+02, 4.74064846e+02, 3.95723101e+02,
                                       3.14365888e+02, 2.47997260e+02, 2.01949742e+02, 1.87939993e+02,
                                       1.47301900e+02, 1.22057347e+02, 9.44857640e+01, 7.59628917e+01,
                                       6.01736445e+01, 4.40971512e+01, 3.45815865e+01, 2.77571337e+01,
                                       2.12604387e+01, 1.78079757e+01, 1.32672151e+01, 1.04391673e+01,
                                       8.40114179e+00, 6.95752329e+00, 6.39158281e+00, 6.01332523e+00,
                                       4.49567083e+00, 3.62891391e+00, 3.11307538e+00, 2.18187534e+00,
                                       1.61615641e+00, 1.75082127e+00, 2.51349128e+00, 2.90669157e+00,
                                       3.03402711e+00, 1.70745712e+00, 9.39018836e-01, 1.36677463e+00,
                                       6.77851414e-01],
                                      rtol=1e-5)
    # test_halfmaps_stats()
# class TestFunctions

class TestSPACommands(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="servaltest_")
        os.chdir(self.wd)
        print("In", self.wd)
    # setUp()
    
    def tearDown(self):
        os.chdir(root)
        shutil.rmtree(self.wd)
    # tearDown()

    def test_fofc(self):
        sys.argv = ["", "fofc", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"], "-d", "1.9"]
        main()
        self.assertTrue(os.path.isfile("diffmap_maps.mtz"))
        mtz = gemmi.read_mtz_file("diffmap_maps.mtz")
        self.assertEqual(mtz.nreflections, 208238)
    # test_fofc()

    def test_fsc(self):
        sys.argv = ["", "fsc", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"], "--mask", data["mask"]]
        main()
        self.assertTrue(os.path.isfile("fsc.dat"))
        df = pandas.read_table("fsc.dat", comment="#", sep=r"\s+")
        
        numpy.testing.assert_allclose(df.fsc_FC_full,
                                      [0.998242, 0.983364, 0.942893, 0.922745, 0.914275, 0.913022,
                                       0.900049, 0.862226, 0.882023, 0.860902, 0.875105, 0.902496,
                                       0.895989, 0.908309, 0.928456, 0.945388, 0.960408, 0.965396,
                                       0.962765, 0.959932, 0.957614, 0.959571, 0.957603, 0.955088,
                                       0.943898, 0.937702, 0.937046, 0.933706, 0.922681, 0.886917,
                                       0.857613, 0.822838, 0.799404, 0.774533, 0.732929, 0.659752,
                                       0.623708, 0.599147, 0.552802, 0.508552, 0.425997, 0.30048,
                                       0.286845, 0.231512, 0.164211, 0.134033, 0.089944, 0.061003,
                                       0.024544, 0.009376, -0.004681, 0.02305, 0.030188],
                                      rtol=0, atol=1e-4)
        numpy.testing.assert_allclose(df.fsc_half_masked_corrected,
                                      [0.99998, 0.999854, 0.999561, 0.998637, 0.998478, 0.998217, 0.998024, 0.996446, 0.995316,
                                       0.992014, 0.991325, 0.992381, 0.990588, 0.989501, 0.990386, 0.991547, 0.992241, 0.99298,
                                       0.991725, 0.989109, 0.983623, 0.976955, 0.970857, 0.96608, 0.958066, 0.947261, 0.925901,
                                       0.896515, 0.865408, 0.833411, 0.781538, 0.754205, 0.703446, 0.653649, 0.587084, 0.513894,
                                       0.459777, 0.393584, 0.349775, 0.333019, 0.260333, 0.231767, 0.187071, 0.150573, 0.108945,
                                       0.102676, 0.140925, 0.134152, 0.136187, 0.104752, 0.063349, 0.080262, 0.016436],
                                      rtol=0, atol=1e-4)
    # test_fsc()

    @unittest.skipUnless(utils.refmac.check_version(), "refmac unavailable")
    def test_refine(self):
        sys.argv = ["", "refine_spa", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"], "--mask_for_fofc", data["mask"],
                    "--trim_fofc_mtz", "--resolution", "1.9", "--ncycle", "5", "--cross_validation"]
        main()
        self.assertTrue(os.path.isfile("refined_fsc.json"))
        self.assertTrue(os.path.isfile("refined.mmcif"))
        self.assertTrue(os.path.isfile("diffmap_maps.mtz"))
        self.assertTrue(os.path.isfile("shifted_refined.log"))
        self.assertTrue(os.path.isfile("refined_expanded.pdb"))

        with open("shifted_refined.log") as ifs:
            fscavgs = [float(l.split()[-1]) for l in ifs if l.startswith("Average Fourier shell correlation")]
        self.assertEqual(len(fscavgs), 6)
        self.assertGreater(fscavgs[-1], 0.6)
        
        st = utils.fileio.read_structure("refined_expanded.pdb")
        self.assertEqual(len(st[0]), 4)

        sys.argv = ["", "util", "json2csv", "refined_fsc.json"]
        main()
        # TODO check result?
    # test_refine()

    def test_trim(self):
        sys.argv = ["", "trim", "--maps", data["half1"], data["half2"],
                    "--model", data["pdb"], "--mask", data["mask"],
                    "--no_shift", "--noncubic", "--noncentered"]
        main()
        self.assertTrue(os.path.isfile("emd_30913_half_map_1_trimmed.mrc"))
        self.assertTrue(os.path.isfile("emd_30913_half_map_2_trimmed.mrc"))
        self.assertTrue(os.path.isfile("emd_30913_msk_1_trimmed.mrc"))
        self.assertTrue(os.path.isfile("trim_shifts.json"))

        with open("trim_shifts.json") as ifs:
            shifts = json.load(ifs)
        numpy.testing.assert_allclose(shifts["shifts"], [-1.59999327, -4.79997982, -7.99996636], rtol=1e-5)
        self.assertEqual(shifts["new_grid"], [106, 98, 90])
    # test_trim()

    def test_translate(self):
        pass

    def test_localcc(self):
        sys.argv = ["", "localcc", "--halfmaps", data["half1"], data["half2"],
                    "--model", data["pdb"], "--mask", data["mask"], "--kernel", "5"]
        main()
        self.assertTrue(os.path.isfile("ccmap_r5px_half.mrc"))
        self.assertTrue(os.path.isfile("ccmap_r5px_model.mrc"))

        st = utils.fileio.read_structure(data["pdb"])
        halfcc = utils.fileio.read_ccp4_map("ccmap_r5px_half.mrc")[0]
        modelcc = utils.fileio.read_ccp4_map("ccmap_r5px_model.mrc")[0]

        self.assertAlmostEqual(numpy.mean([modelcc.interpolate_value(cra.atom.pos) for cra in st[0].all()]),
                               0.6416836618309301, places=3)
        self.assertAlmostEqual(numpy.mean([halfcc.interpolate_value(cra.atom.pos) for cra in st[0].all()]),
                               0.6619259582976047, places=3)

    def test_commands(self): # util commands
        sys.argv = ["", "util", "symmodel", "--model", data["pdb"], "--map", data["mask"],
                    "--pg", "D2", "--biomt"]
        main()

        sys.argv = ["", "util", "expand", "--model", "pdb7dy0_asu.pdb"]
        main()

        sys.argv = ["", "util", "h_add", data["pdb"]]
        main()

        # TODO merge_models

        sys.argv = ["", "util", "power", "--map", data["mask"], data["half1"], data["half2"]]
        main()

        sys.argv = ["", "util", "fcalc", "--model", data["pdb"], "-d", "1.7", "--auto_box_with_padding=5"]
        main()

        sys.argv = ["", "util", "nemap", "--halfmaps", data["half1"], data["half2"],
                    "--mask", data["mask"], "--trim_mtz", "-d", "1.7"]
        main()

        sys.argv = ["", "util", "blur", "--hklin", "nemap_maps.mtz", "-B", "100"]
        main()
    # test_commands()
# class TestSPACommands

if __name__ == '__main__':
    unittest.main()

