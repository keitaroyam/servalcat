"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.xtal.french_wilson import process_input, determine_Sigma_and_aniso, french_wilson
from servalcat import utils
import argparse
import subprocess
import gemmi
import time
import plotly.express as px

def add_arguments(parser):
    parser.add_argument('--hklin', required=True,
                        help='Input MTZ file')
    parser.add_argument('--labin', required=True,
                        help='MTZ column for I,SIGI')
    parser.add_argument('--d_min', type=float)
    parser.add_argument('--d_max', type=float)
    parser.add_argument('--nbins', type=int, default=20,
                        help="Number of bins (default: %(default)d)")
    parser.add_argument('-o','--output_prefix', default="servalcat_fw",
                        help='output file name prefix (default: %(default)s)')
# add_arguments()

def run_ctruncate(hklin, labin):
    hklout = "ctruncate.mtz"
    cmd = ["ctruncate", "-hklin", hklin, "-colin", "/*/*/[{}]".format(labin), "-hklout", hklout]
    #ofs = open("ctruncate.log", "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    read_mat = False
    mat = []
    for l in iter(p.stdout.readline, ""):
        if "Anisotropic B scaling (orthogonal coords):" in l:
            read_mat = True
        elif read_mat and l.strip():
            mat.append([float(x) for x in l.split()[1:4]])
            if len(mat) == 3:
                read_mat = False
    print("exit with", p.wait())
    t = time.time() - t0
    B = gemmi.SMat33d(mat[0][0], mat[1][1], mat[2][2], mat[0][1], mat[0][2], mat[1][2])
    b_iso = B.trace() / 3
    b_aniso = B.added_kI(-b_iso)
    mtz = gemmi.read_mtz_file(hklout)
    hkldata = utils.hkl.hkldata_from_mtz(mtz, ["F", "SIGF"], newlabels=["F_ct", "SIGF_ct"], require_types=["F", "Q"])
    return b_aniso, hkldata, t

def main(args):
    B_ct, hkldata_ct, t_ct = run_ctruncate(args.hklin, args.labin)
    t0 = time.time()
    hkldata, _, _, _ = process_input(hklin=args.hklin,
                                     labin=args.labin.split(","),
                                     n_bins=args.nbins,
                                     free=None,
                                     xyzins=[],
                                     source=None,
                                     d_min=args.d_min)
    B_aniso = determine_Sigma_and_aniso(hkldata)
    french_wilson(hkldata, B_aniso)
    t_se = time.time() - t0

    hkldata.merge(hkldata_ct.df)
    hkldata.df["snr"] = hkldata.df.I / hkldata.df.SIGI
    print(hkldata.df)

    fig = px.scatter(hkldata.df, x="F_ct", y="F", symbol="centric",
                     color="snr",
                     facet_col='bin', facet_col_wrap=4,
                     render_mode='webgl',
                     #log_y=True, log_x=True,
                     trendline="ols", trendline_scope="overall")
    #fig.update_yaxes(matches=None)
    #fig.update_xaxes(matches=None)
    fig.show()
    #results = px.get_trendline_results(fig)
    #print(results)
    fig = px.scatter(hkldata.df, x="SIGF_ct", y="SIGF",  symbol="centric", render_mode='webgl',
                     facet_col='bin', facet_col_wrap=4,
                     color="snr",
                     #log_y=True, log_x=True,
                     trendline="ols", trendline_scope="overall")
    fig.show()

    print("B_aniso")
    print("serval:   ", B_aniso)
    print("ctruncate:", B_ct)
    print()
    print("Time")
    print("serval:   ", t_se)
    print("ctruncate:", t_ct)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
