# Servalcat
**S**tructur**e** **r**efinement and **val**idation for **c**rystallography and single p**a**r**t**icle analysis

Currently we focus on development for single particle analysis and there are only a few functions at the moment.

## Installation
```
pip install git+https://github.com/project-gemmi/gemmi.git
pip install git+https://github.com/keitaroyam/servalcat.git
```
Add `-U` option for updating. Servalcat often requires new [GEMMI](https://github.com/project-gemmi/gemmi) features (not the latest from pypi, but from github).

The required GEMMI version is now [v0.4.6-2-g58e6395c](https://github.com/project-gemmi/gemmi/commit/58e6395c95f92a565d039074ecdd862331a50ef8). Please update GEMMI as well if it is old.

## Usage

### Refinement using REFMAC5
Servalcat makes refinement by REFMAC5 easy for single particle analysis. The weighted and sharpened Fo-Fc map is calculated after the refinement. For details please see the reference.

Make a new directory and run:
```
servalcat refine_spa --model input.pdb --resolution 2.5 --halfmaps ../half_map_1.map ../half_map_2.map --ncycle 10 [--pg C2]
```
Specify unsharpened and unweighted half maps (e.g. those after Refine3D of RELION) after `--halfmaps`.

If map has been symmetrised with a point group, asymmetric unit model should be given together with `--pg` to specify a point group symbol.
It assumes the center of the box is the origin of the symmetry and the axis convention follows RELION.

Useful options:
- `--mask_for_fofc mask.mrc` : speify mask file for Fo-Fc map calculation
- `--jellybody` : turn on jelly body refinment
- `--weight_auto_scale value` : specify weight auto scale. by default Servalcat determines it from resolution and mask/box ratio
- `--keyword_file file` : specify any refmac keyword file(s) (e.g. prosmart restraint file)
- `--pixel_size value` : override pixel size of map
- `--exe refmac5` : specify REFMAC5 binary

Output files:
- `refined.pdb`: refined model
- `refined_expanded.pdb`: symmetry-expanded version
- `diffmap.mtz`: can be auto-opened with coot. sharpened and weighted Fo map and Fo-Fc map
- `diffmap_normalized_fofc.mrc`: Fo-Fc map normalized within a mask. Look at raw values
- `local_refined.log`: refmac log file

### Map trimming
Maps from single particle analysis often have very large size due to unnccesary region outside the molecule. You can save disk space by trimming the unnccesary region.
```
servalcat trim --maps postprocess.mrc halfmap1.mrc halfmap2.mrc [--mask mask.mrc] [--model model.pdb] [--padding 10]
```
Maps specified with `--maps` are trimed. The boundary is decided by `--mask` or `--model` if mask is not available.
Model(s) are shifted into a new box.
By default new boundary is centered on the original map and cubic, but they can be turned off with `--noncentered` and `--noncubic`.

## Reference
[Yamashita, K., Palmer, C. M., Burnley, T., Murshudov, G. N. (2021) "Cryo-EM single particle structure refinement and map calculation using Servalcat" *bioRxiv*](https://doi.org/10.1101/2021.05.04.442493)
