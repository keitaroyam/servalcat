# Servalcat
**S**tructur**e** **r**efinement and **val**idation for **c**rystallography and single p**a**r**t**icle analysis

Currently we focus on development for single particle analysis and there are only a few functions at the moment.

## Installation

```
pip install servalcat
```
will install the stable version.

If you want to try the latest features,
```
pip install git+https://github.com/keitaroyam/servalcat.git
```
Add `-U` option for updating.

The required GEMMI version is now [v0.5.7](https://github.com/project-gemmi/gemmi/releases/tag/v0.5.7). Please update GEMMI as well if it is old (it should be done by pip).

## Usage
```
servalcat <command> <args>
```
The most useful `command`s are shown below. To see all arguments for each `command` please run
```
servalcat <command> -h
```

Please also read the documentation: https://servalcat.readthedocs.io/en/latest/

### Refinement using REFMAC5
Servalcat makes refinement by REFMAC5 easy for single particle analysis. The weighted and sharpened Fo-Fc map is calculated after the refinement. For details please see the reference.

Make a new directory and run:
```
servalcat refine_spa \
 --model input.pdb --resolution 2.5 \
 --halfmaps ../half_map_1.mrc ../half_map_2.mrc \
 --ncycle 10 [--pg C2] \
 [--mask_for_fofc mask.mrc]
```
Specify unsharpened and unweighted half maps (e.g. those after Refine3D of RELION) after `--halfmaps`.

If map has been symmetrised with a point group, asymmetric unit model should be given together with `--pg` to specify a point group symbol.
It assumes the center of the box is the origin of the symmetry and the axis convention follows [RELION](https://relion.readthedocs.io/en/latest/Reference/Conventions.html#symmetry).

Other useful options:
- `--ligand lig.cif` : specify restraint dictionary (.cif) file(s)
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
- `shifted_refined.log`: refmac log file

### Fo-Fc map calculation
It is important to refine individual atomic B values with electron scattering factors to calculate meaningful Fo-Fc map.
Fo-Fc map is calculated in `refine_spa` command (explained above) so usually you do not need to run `fofc` command manually, but you may want to calculate e.g. omit maps.
```
servalcat fofc \
 --model input.pdb --resolution 2.5 \
 --halfmaps ../half_map_1.mrc ../half_map_2.mrc \
 [--mask mask.mrc] [-o output_prefix] [-B B value]
```

`-B` is to calculate weighted maps based on local B estimate. It may be useful for model building in noisy region.

### Map trimming
Maps from single particle analysis often have very large size due to unnccesary region outside the molecule. You can save disk space by trimming the unnccesary region.
```
servalcat trim \
 --maps postprocess.mrc halfmap1.mrc halfmap2.mrc \
 [--mask mask.mrc] [--model model.pdb] [--padding 10]
```
Maps specified with `--maps` are trimed. The boundary is decided by `--mask` or `--model` if mask is not available.
Model(s) are shifted into a new box.
By default new boundary is centered on the original map and cubic, but they can be turned off with `--noncentered` and `--noncubic`.
If you do not want to shift maps and models, specify `--no_shift` to keep origin.

## Reference
[Yamashita, K., Palmer, C. M., Burnley, T., Murshudov, G. N. (2021) "Cryo-EM single particle structure refinement and map calculation using Servalcat" *Acta Cryst. D***77**, 1282-1291](https://doi.org/10.1107/S2059798321009475)
