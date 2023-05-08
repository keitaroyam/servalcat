import sys
import json
import pandas
import gemmi

csv_in, json_out = sys.argv[1:]
df = pandas.read_csv(csv_in)
ret = {}
for x in df.to_dict(orient="records"):
    metal = gemmi.Element(x["Metal"]).name
    ligand = gemmi.Element(x["Ligand"]).name
    l = ret.setdefault(metal, {}).setdefault(ligand, [])
    modes = []
    for i in range(1, 4):
        m = x["mode{}".format(i)]
        s = x["std{}".format(i)]
        if m == m and s == s: # not nan
            modes.append({"mode": m, "std": s})
    l.append({"coord": x["Coordination"],
              "median": x["median"],
              "mad": x["mad"],
              "mean": x["mean"],
              "std": x["std"] if x["std"] == x["std"] else 0,
              "count": x["count"],
              "modes": modes})

json.dump({"version": 1, "date": "2023-05-05", "metal_coordination":ret}, open(json_out, "w"), indent=1, allow_nan=False)

"""
{'Metal': 'Ba',
 'Ligand': 'Cl',
 'Coordination': 8,
 'median': 3.12194718739944,
 'mad': 0.0398797514650128,
 'mean': 3.1440235683357343,
 'std': 0.0460297586743337,
 'count': 20,
 'nmodes': 0,
 'mode1': nan,
 'std1': nan,
 'mode2': nan,
 'std2': nan,
 'mode3': nan,
 'std3': nan}
"""
