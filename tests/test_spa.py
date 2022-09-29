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
from servalcat import command_line

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

    url_md5 = (('half1', 'https://ftp.wwpdb.org/pub/emdb/structures/EMD-30913/other/emd_30913_half_map_1.map.gz', '0bf81c0057d3017972e7cf53e8a8df77'),
               ('half2', 'https://ftp.wwpdb.org/pub/emdb/structures/EMD-30913/other/emd_30913_half_map_2.map.gz', '5bba9c32a95d6675fe0387342af5a439'),
               ('mask', 'https://ftp.wwpdb.org/pub/emdb/structures/EMD-30913/masks/emd_30913_msk_1.map', 'e69403cec5be349b097c8080ce5bb1fd'),
               ('pdb', 'https://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/dy/pdb7dy0.ent.gz', '78351986de7c8a0dacef823dd39f6f8b'),
               ('mmcif', 'https://ftp.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/dy/7dy0.cif.gz', '107d35d6f214cf330c63c2da0d1a63d8'))

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
        self.assertEqual(st[0].count_hydrogen_sites(), 0)
        monlib = utils.restraints.load_monomer_library(st)
        utils.restraints.add_hydrogens(st, monlib, "elec")
        self.assertEqual(st[0].count_hydrogen_sites(), 841)
        bad_h = [numpy.allclose(cra.atom.pos.tolist(), [0,0,0]) for cra in st[0].all()]
        self.assertEqual(sum(bad_h), 0)
    # test_h_add()

    def test_halfmaps_stats(self):
        maps = load_halfmaps()
        mask = load_mask()
        d_min = utils.maps.nyquist_resolution(maps[0][0])
        self.assertAlmostEqual(d_min, 1.6, places=3)
        hkldata = utils.maps.mask_and_fft_maps(maps, d_min, mask)
        hkldata.setup_relion_binning()
        utils.maps.calc_noise_var_from_halfmaps(hkldata)
        self.assertTrue(numpy.allclose(hkldata.binned_df.var_signal, [3.01280911e+04, 2.73253902e+03, 1.12831493e+03, 7.78811837e+02,
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
                                                                      6.77851414e-01]))
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
        sys.argv = ["", "fofc", "--halfmaps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--model", pipes.quote(data["pdb"]), "-d", "1.9"]
        command_line.main()
        self.assertTrue(os.path.isfile("diffmap.mtz"))
        mtz = gemmi.read_mtz_file("diffmap.mtz")
        self.assertEqual(mtz.nreflections, 208238)
    # test_fofc()

    def test_fsc(self):
        sys.argv = ["", "fsc", "--halfmaps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--model", pipes.quote(data["pdb"]), "--mask", pipes.quote(data["mask"])]
        command_line.main()
        self.assertTrue(os.path.isfile("fsc.dat"))
        df = pandas.read_table("fsc.dat", comment="#", sep="\s+")
        self.assertTrue(numpy.allclose(df.fsc_FC_full, [ 0.999384,  0.982716,  0.94365 ,  0.923566,  0.91484 ,  0.913189,
                                                         0.900186,  0.8624  ,  0.882143,  0.861254,  0.875493,  0.902729,
                                                         0.896093,  0.908346,  0.928542,  0.945463,  0.9605  ,  0.96549 ,
                                                         0.96293 ,  0.960066,  0.957779,  0.959628,  0.957602,  0.955046,
                                                         0.943849,  0.937653,  0.936928,  0.933555,  0.922384,  0.886468,
                                                         0.856839,  0.821977,  0.798375,  0.773678,  0.731655,  0.657545,
                                                         0.622479,  0.597957,  0.55077 ,  0.507593,  0.423219,  0.298562,
                                                         0.285342,  0.229713,  0.163543,  0.133819,  0.088876,  0.059217,
                                                         0.023549,  0.008861, -0.005151,  0.023599,  0.030168]))
        self.assertTrue(numpy.allclose(df.fsc_half, [0.99998 , 0.999857, 0.999549, 0.998618, 0.998465, 0.998222,
                                                     0.998039, 0.996382, 0.995207, 0.991879, 0.991197, 0.992273,
                                                     0.990446, 0.989377, 0.990265, 0.99144 , 0.992204, 0.992934,
                                                     0.991666, 0.98904 , 0.983504, 0.976701, 0.970605, 0.965812,
                                                     0.95771 , 0.946897, 0.92524 , 0.902904, 0.872943, 0.839902,
                                                     0.793809, 0.765165, 0.71621 , 0.667087, 0.588015, 0.511396,
                                                     0.459723, 0.386018, 0.347216, 0.3433  , 0.273476, 0.228981,
                                                     0.193635, 0.131909, 0.098189, 0.104355, 0.147002, 0.162024,
                                                     0.15678 , 0.087615, 0.049143, 0.075087, 0.04238 ]))
    # test_fsc()

    #@unittest.skip("skip refmac")
    def test_refine(self):
        sys.argv = ["", "refine_spa", "--halfmaps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--model", pipes.quote(data["pdb"]), "--mask_for_fofc", pipes.quote(data["mask"]),
                    "--no_shift", "--trim_fofc_mtz", "--resolution", "1.9", "--ncycle", "5", "--cross_validation"]
        command_line.main()
        self.assertTrue(os.path.isfile("refined_fsc.json"))
        self.assertTrue(os.path.isfile("refined.mmcif"))
        self.assertTrue(os.path.isfile("diffmap.mtz"))
        self.assertTrue(os.path.isfile("shifted_refined.log"))
        self.assertTrue(os.path.isfile("refined_expanded.pdb"))

        fscavgs = [float(l.split()[-1]) for l in open("shifted_refined.log") if l.startswith("Average Fourier shell correlation")]
        self.assertEqual(len(fscavgs), 6)
        self.assertGreater(fscavgs[-1], 0.6)
        
        st = utils.fileio.read_structure("refined_expanded.pdb")
        self.assertEqual(len(st[0]), 4)

        sys.argv = ["", "util", "json2csv", "refined_fsc.json"]
        command_line.main()
        # TODO check result?
    # test_refine()

    def test_trim(self):
        sys.argv = ["", "trim", "--maps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--model", pipes.quote(data["pdb"]), "--mask", pipes.quote(data["mask"]),
                    "--no_shift", "--noncubic", "--noncentered"]
        command_line.main()
        self.assertTrue(os.path.isfile("emd_30913_half_map_1_trimmed.mrc"))
        self.assertTrue(os.path.isfile("emd_30913_half_map_2_trimmed.mrc"))
        self.assertTrue(os.path.isfile("emd_30913_msk_1_trimmed.mrc"))
        self.assertTrue(os.path.isfile("trim_shifts.json"))

        shifts = json.load(open("trim_shifts.json"))
        self.assertTrue(numpy.allclose(shifts["shifts"], [-1.59999327, -4.79997982, -7.99996636]))
        self.assertEqual(shifts["new_grid"], [106, 98, 90])
    # test_trim()

    def test_translate(self):
        pass

    def test_localcc(self):
        sys.argv = ["", "localcc", "--halfmaps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--model", pipes.quote(data["pdb"]), "--mask", pipes.quote(data["mask"]), "--kernel", "5"]
        command_line.main()
        self.assertTrue(os.path.isfile("ccmap_r5px_half.mrc"))
        self.assertTrue(os.path.isfile("ccmap_r5px_model.mrc"))

        st = utils.fileio.read_structure(data["pdb"])
        halfcc = utils.fileio.read_ccp4_map("ccmap_r5px_half.mrc")[0]
        modelcc = utils.fileio.read_ccp4_map("ccmap_r5px_model.mrc")[0]

        self.assertAlmostEqual(numpy.mean([modelcc.interpolate_value(cra.atom.pos) for cra in st[0].all()]),
                               0.6416836618309301, places=4)
        self.assertAlmostEqual(numpy.mean([halfcc.interpolate_value(cra.atom.pos) for cra in st[0].all()]),
                               0.6619259582976047, places=4)

    def test_commands(self): # util commands
        sys.argv = ["", "util", "symmodel", "--model", pipes.quote(data["pdb"]), "--map", pipes.quote(data["mask"]),
                    "--pg", "D2", "--biomt"]
        command_line.main()

        sys.argv = ["", "util", "expand", "--model", "pdb7dy0_asu.pdb"]
        command_line.main()

        sys.argv = ["", "util", "h_add", pipes.quote(data["pdb"])]
        command_line.main()

        # TODO merge_models

        sys.argv = ["", "util", "power", "--map", pipes.quote(data["mask"]), pipes.quote(data["half1"]), pipes.quote(data["half2"])]
        command_line.main()

        sys.argv = ["", "util", "fcalc", "--model", pipes.quote(data["pdb"]), "-d", "1.7", "--auto_box_with_padding=5"]
        command_line.main()

        sys.argv = ["", "util", "nemap", "--halfmaps", pipes.quote(data["half1"]), pipes.quote(data["half2"]),
                    "--mask", pipes.quote(data["mask"]), "--trim_mtz", "-d", "1.7"]
        command_line.main()

        sys.argv = ["", "util", "blur", "--hklin", "nemap.mtz", "-B", "100"]
        command_line.main()
    # test_commands()
# class TestSPACommands

if __name__ == '__main__':
    unittest.main()

